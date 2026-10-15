using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Management;
using System.Reflection;
using System.Security.Principal;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;

public sealed class InstallOptions {
    public string Target=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),"HopBox-Infer"), ProgressFile;
    public bool Quiet, Launch=true;
}
public static class Setup {
    public static string NormalizeTarget(string value) {
        if(String.IsNullOrWhiteSpace(value)||!Path.IsPathRooted(value)||value.StartsWith("\\\\")) throw new ArgumentException("Choose an absolute folder on a local drive.");
        string target=Path.GetFullPath(value).TrimEnd('\\','/');
        if(Path.GetPathRoot(target).TrimEnd('\\')==target||target.Length<8||File.Exists(target)||target.IndexOf('"')>=0) throw new ArgumentException("Choose a dedicated application folder, rather than a drive root or file.");
        foreach(var special in new[]{Environment.SpecialFolder.Windows,Environment.SpecialFolder.System,Environment.SpecialFolder.ProgramFiles,Environment.SpecialFolder.ProgramFilesX86,Environment.SpecialFolder.UserProfile,Environment.SpecialFolder.DesktopDirectory,Environment.SpecialFolder.MyDocuments,Environment.SpecialFolder.LocalApplicationData,Environment.SpecialFolder.ApplicationData,Environment.SpecialFolder.CommonApplicationData}) {
            string protectedFolder=Environment.GetFolderPath(special).TrimEnd('\\');
            if(!String.IsNullOrEmpty(protectedFolder)&&String.Equals(target,protectedFolder,StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("Choose a dedicated HopBox-Infer subfolder inside that location.");
        }
        bool existingApp=File.Exists(Path.Combine(target,"HopBox-Infer.exe"))&&File.Exists(Path.Combine(target,"python","pythonw.exe"))&&Directory.Exists(Path.Combine(target,"app"));
        if(Directory.Exists(target)&&Directory.GetFileSystemEntries(target).Length!=0&&!existingApp) throw new ArgumentException("This folder contains other files. Choose an empty folder or an existing HopBox Infer installation.");
        return target;
    }
    static void MoveWithRetry(string source,string destination) {
        for(int attempt=0;;attempt++) {
            try { Directory.Move(source,destination); return; }
            catch(IOException) { if(attempt==29) throw; }
            catch(UnauthorizedAccessException) { if(attempt==29) throw; }
            Thread.Sleep(500);
        }
    }
    static void Extract(string resource,string prefix,string destination,Action<int,string> update,int from,int to) {
        Directory.CreateDirectory(destination);
        using(var payload=Assembly.GetExecutingAssembly().GetManifestResourceStream(resource)) {
            if(payload==null) throw new IOException("This installer does not include the selected component.");
            using(var zip=new ZipArchive(payload,ZipArchiveMode.Read)) {
                long total=0,done=0; foreach(var entry in zip.Entries) total+=entry.Length;
                foreach(var entry in zip.Entries) {
                    string relative=entry.FullName.Replace('/','\\');
                    if(!relative.StartsWith(prefix,StringComparison.Ordinal)) throw new IOException("Unexpected package root.");
                    string output=Path.GetFullPath(Path.Combine(destination,relative.Substring(prefix.Length)));
                    if(!output.StartsWith(destination+"\\",StringComparison.OrdinalIgnoreCase)) throw new IOException("Unsafe package member.");
                    if(entry.Name.Length==0) { Directory.CreateDirectory(output); continue; }
                    Directory.CreateDirectory(Path.GetDirectoryName(output));
                    using(var source=entry.Open()) using(var target=File.Create(output)) source.CopyTo(target);
                    done+=entry.Length;
                    update(from+(int)((to-from)*done/Math.Max(1,total)),"Installing core and DirectML…");
                }
            }
        }
    }
    public static int Install(InstallOptions options,Action<int,string> progress) {
        string staging=null,backup=null,target=null,phase="checking the destination"; bool committed=false;
        Action<int,string> update=delegate(int p,string t) { progress(p,t); if(!String.IsNullOrEmpty(options.ProgressFile)) try { File.WriteAllText(options.ProgressFile,p+"\n"+t); } catch {} };
        try {
            target=NormalizeTarget(options.Target);
            string parent=Path.GetDirectoryName(target); Directory.CreateDirectory(parent);
            if(new DriveInfo(Path.GetPathRoot(target)).AvailableFreeSpace<1400000000L) throw new IOException("There is not enough free space for extraction and an upgrade backup.");
            phase="extracting the new installation";
            staging=Path.Combine(parent,".hopbox-install-"+Guid.NewGuid().ToString("N")); Directory.CreateDirectory(staging);
            Extract("HopBoxPayload","HopBox-Infer\\",staging,update,0,95);
            if(!File.Exists(Path.Combine(staging,"HopBox-Infer.exe"))||!File.Exists(Path.Combine(staging,"app","hopbox-code.zip"))) throw new IOException("The package is incomplete.");
            File.WriteAllText(Path.Combine(staging,"installation.json"),new JavaScriptSerializer().Serialize(new { version=Branding.Version,components="core",installed_utc=DateTime.UtcNow.ToString("o"),destination=target }));
            if(Directory.Exists(target)) {
                phase="preserving the previous installation";
                string candidate=target+".previous-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss")+"-"+Guid.NewGuid().ToString("N").Substring(0,6);
                MoveWithRetry(target,candidate); backup=candidate;
            }
            phase="activating the new installation"; update(97,"Finishing installation…");
            MoveWithRetry(staging,target); staging=null; committed=true; update(100,"Installation complete.");
            if(options.Launch) Launch(target);
            return 0;
        } catch(Exception error) {
            string recovery="";
            if(!committed&&backup!=null&&Directory.Exists(backup)&&!Directory.Exists(target)) {
                try { MoveWithRetry(backup,target); recovery="The previous installation was restored."; }
                catch(Exception restore) { recovery="The previous installation is preserved at "+backup+". Restore failed: "+restore.Message; }
            }
            string detail="Installation failed while "+phase+".\r\n"+error+"\r\n"+recovery;
            try { if(target!=null) File.WriteAllText(target+".installation-error.txt",detail); } catch {}
            try { File.WriteAllText(Path.Combine(Path.GetTempPath(),"HopBox-Infer-installation-error.txt"),detail); } catch {}
            update(-1,"Installation failed while "+phase+".\r\n"+error.Message+"\r\n"+recovery);
            // Only this process's fresh staging tree can be removed here.
            if(staging!=null&&Directory.Exists(staging)) try { Directory.Delete(staging,true); } catch {}
            return 1;
        }
    }
    public static void Launch(string target) { Process.Start(new ProcessStartInfo(Path.Combine(target,"HopBox-Infer.exe")) { UseShellExecute=true }); }
    public static bool NeedsElevation(string target) {
        using(var identity=WindowsIdentity.GetCurrent()) if(new WindowsPrincipal(identity).IsInRole(WindowsBuiltInRole.Administrator)) return false;
        string parent=Path.GetDirectoryName(target); while(!Directory.Exists(parent)) parent=Path.GetDirectoryName(parent);
        string probe=Path.Combine(parent,".hopbox-permission-"+Guid.NewGuid().ToString("N"));
        try { using(File.Create(probe)) {} File.Delete(probe); return false; } catch(UnauthorizedAccessException) { return true; }
    }
    public static void Shortcuts(string target,bool desktop) {
        var type=Type.GetTypeFromProgID("WScript.Shell"); var shell=Activator.CreateInstance(type);
        string folder=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs),"HopBox Infer"); Directory.CreateDirectory(folder);
        SaveShortcut(type,shell,Path.Combine(folder,"HopBox Infer.lnk"),Path.Combine(target,"HopBox-Infer.exe"),target);
        if(desktop) SaveShortcut(type,shell,Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory),"HopBox Infer.lnk"),Path.Combine(target,"HopBox-Infer.exe"),target);
    }
    static void SaveShortcut(Type type,object shell,string path,string exe,string root) {
        object link=type.InvokeMember("CreateShortcut",BindingFlags.InvokeMethod,null,shell,new object[]{path}); Type lt=link.GetType();
        lt.InvokeMember("TargetPath",BindingFlags.SetProperty,null,link,new object[]{exe}); lt.InvokeMember("WorkingDirectory",BindingFlags.SetProperty,null,link,new object[]{root});
        lt.InvokeMember("IconLocation",BindingFlags.SetProperty,null,link,new object[]{exe+",0"}); lt.InvokeMember("Save",BindingFlags.InvokeMethod,null,link,null);
    }
    [STAThread] public static int Main(string[] args) {
        var options=new InstallOptions(); string render=null;
        try {
            if(!Environment.Is64BitOperatingSystem) throw new PlatformNotSupportedException("HopBox Infer requires 64-bit Windows 10 or 11.");
            bool explicitTarget=false;
            for(int i=0;i<args.Length;i++) {
                if(args[i]=="--install-dir"&&i+1<args.Length) { options.Target=args[++i]; explicitTarget=true; }
                else if(args[i]=="--components"&&i+1<args.Length) { string value=args[++i]; if(value!="core") throw new ArgumentException("The validated component is core (CPU and DirectML for compatible GPUs)."); }
                else if(args[i]=="--no-launch") options.Launch=false;
                else if(args[i]=="--quiet") options.Quiet=true;
                else if(args[i]=="--progress-file"&&i+1<args.Length) options.ProgressFile=args[++i];
                else if(args[i]=="--render-wizard"&&i+1<args.Length) render=args[++i];
                else throw new ArgumentException("Unknown installer argument: "+args[i]);
            }
            if(options.Quiet&&!explicitTarget) throw new ArgumentException("Quiet installation requires --install-dir and an administrator-approved destination.");
            if(options.Quiet) return Install(options,delegate(int p,string t) {});
            Application.EnableVisualStyles(); Application.SetCompatibleTextRenderingDefault(false);
            using(var wizard=new InstallWizard(options)) {
                if(render!=null) { wizard.Show(); Application.DoEvents(); using(var bitmap=new Bitmap(wizard.Width,wizard.Height)) { wizard.DrawToBitmap(bitmap,new Rectangle(Point.Empty,bitmap.Size)); bitmap.Save(render); } return 0; }
                Application.Run(wizard); return wizard.Result;
            }
        } catch(Exception error) {
            try { File.WriteAllText(Path.Combine(Path.GetTempPath(),"HopBox-Infer-installation-error.txt"),error.ToString()); } catch {}
            if(!options.Quiet) MessageBox.Show(error.Message,Branding.Name+" installation",MessageBoxButtons.OK,MessageBoxIcon.Error); return 1;
        }
    }
}

public sealed class InstallWizard : Form {
    readonly InstallOptions options; readonly TextBox destination; readonly CheckBox desktop,open;
    readonly Button install,cancel,browse; readonly Label status,adapters; readonly ProgressBar progress;
    bool busy,finished; public int Result=2;
    public InstallWizard(InstallOptions options) {
        this.options=options; Text="HopBox Infer Setup — "+Branding.Version; Icon=Icon.ExtractAssociatedIcon(Assembly.GetExecutingAssembly().Location);
        StartPosition=FormStartPosition.CenterScreen; FormBorderStyle=FormBorderStyle.FixedDialog; MaximizeBox=false; AutoScaleMode=AutoScaleMode.Dpi; ClientSize=new Size(660,610);
        Font=new Font("Segoe UI",10); BackColor=Color.FromArgb(247,249,247); ForeColor=Color.FromArgb(23,66,53);
        var logo=new PictureBox { Bounds=new Rectangle(30,24,68,68),SizeMode=PictureBoxSizeMode.Zoom };
        using(var stream=Assembly.GetExecutingAssembly().GetManifestResourceStream("HopBoxLogo")) using(var image=Image.FromStream(stream)) logo.Image=new Bitmap(image); Controls.Add(logo);
        AddLabel("Install HopBox Infer",116,28,505,38,20,FontStyle.Bold); AddLabel("Version "+Branding.Version+"  ·  Offline Windows x64",118,70,480,24,10,FontStyle.Regular);
        AddLabel("Installation folder",30,124,580,24,11,FontStyle.Bold);
        destination=new TextBox { Text=options.Target,Bounds=new Rectangle(30,157,490,28),AccessibleName="Installation folder" }; Controls.Add(destination);
        browse=new Button { Text="Browse…",Bounds=new Rectangle(532,155,98,32) }; Controls.Add(browse);
        browse.Click+=delegate {
            try {
                string initial;
                try { initial=Path.GetFullPath(destination.Text); }
                catch(ArgumentException) { initial=Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles); }
                catch(NotSupportedException) { initial=Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles); }
                while(!String.IsNullOrEmpty(initial)&&!Directory.Exists(initial)) initial=Path.GetDirectoryName(initial);
                using(var picker=new FolderBrowserDialog { Description="Choose a location or an existing HopBox Infer folder. HopBox-Infer is added inside a nonempty parent folder.",SelectedPath=initial??"",ShowNewFolderButton=true }) {
                    if(picker.ShowDialog(this)!=DialogResult.OK) return;
                    string selected=picker.SelectedPath;
                    destination.Text=File.Exists(Path.Combine(selected,"HopBox-Infer.exe"))||Directory.GetFileSystemEntries(selected).Length==0?selected:Path.Combine(selected,"HopBox-Infer");
                }
            } catch(Exception error) { MessageBox.Show(this,"Could not use that folder: "+error.Message,Text,MessageBoxButtons.OK,MessageBoxIcon.Warning); }
        };
        AddLabel("Choose a location approved by your administrator. Program Files may require\nadministrator consent. User data is stored separately; the app runs as your user.",30,197,600,48,9,FontStyle.Regular);
        AddLabel("Components",30,264,580,25,11,FontStyle.Bold); AddLabel("✓  Core, CPU + DirectML  (required, about 584 MB)",30,298,600,27,10,FontStyle.Bold);
        AddLabel("For CPU and compatible Intel, AMD Radeon or NVIDIA DirectX 12 GPUs.",53,329,577,25,9,FontStyle.Regular);
        adapters=AddLabel("Checking graphics adapters...",53,365,577,72,9,FontStyle.Regular);
        desktop=new CheckBox { Text="Create a desktop shortcut",Bounds=new Rectangle(30,454,305,28),Checked=true }; Controls.Add(desktop);
        open=new CheckBox { Text="Open HopBox Infer when finished",Bounds=new Rectangle(30,486,600,28),Checked=options.Launch }; Controls.Add(open);
        progress=new ProgressBar { Bounds=new Rectangle(30,529,600,6),Minimum=0,Maximum=100 }; Controls.Add(progress);
        status=AddLabel("Ready to install. No Internet connection needed.",30,548,407,46,9,FontStyle.Regular);
        cancel=new Button { Text="Cancel",Bounds=new Rectangle(438,551,88,34) }; Controls.Add(cancel);
        install=new Button { Text="Install",Bounds=new Rectangle(538,551,92,34),BackColor=Color.FromArgb(23,76,60),ForeColor=Color.White,FlatStyle=FlatStyle.Flat }; Controls.Add(install);
        AcceptButton=install; CancelButton=cancel; cancel.Click+=delegate { Close(); }; install.Click+=delegate { if(finished) Close(); else BeginInstall(); };
        FormClosing+=delegate(object sender,FormClosingEventArgs e) { if(busy) e.Cancel=true; };
        Shown+=delegate { new Thread(DetectAdapters) { IsBackground=true }.Start(); };
    }
    Label AddLabel(string text,int x,int y,int width,int height,float size,FontStyle style) { var label=new Label { Text=text,Bounds=new Rectangle(x,y,width,height),Font=new Font("Segoe UI",size,style) }; Controls.Add(label); return label; }
    void DetectAdapters() {
        string text="Adapter detection unavailable. GPU support depends on the installed driver.";
        try {
            var names=new System.Collections.Generic.List<string>();
            using(var search=new ManagementObjectSearcher("SELECT Name FROM Win32_VideoController")) using(var items=search.Get()) foreach(var item in items) { string name=Convert.ToString(item["Name"]); names.Add(name); }
            text="Detected: "+String.Join(", ",names.ToArray())+". "+"CPU and DirectML are included. No vendor-specific GPU pack is needed. Choose a GPU or CPU inside HopBox Infer.";
        } catch {}
        try { BeginInvoke((Action)delegate { if(!busy&&!finished) adapters.Text=text; }); } catch {}
    }
    void UpdateProgress(int percent,string text) { try { BeginInvoke((Action)delegate { progress.Value=Math.Max(0,Math.Min(100,percent)); status.Text=text; }); } catch {} }
    void BeginInstall() {
        bool elevated;
        try { options.Target=Setup.NormalizeTarget(destination.Text); options.Launch=false; elevated=Setup.NeedsElevation(options.Target); }
        catch(Exception error) { MessageBox.Show(this,error.Message,Text,MessageBoxButtons.OK,MessageBoxIcon.Warning); return; }
        busy=true; destination.Enabled=browse.Enabled=desktop.Enabled=open.Enabled=install.Enabled=cancel.Enabled=false;
        status.Text=elevated?"Approve Windows administrator consent to install in this folder…":"Installing…";
        new Thread(delegate() {
            int result=1;
            try {
                if(elevated) {
                    string progressFile=Path.Combine(Path.GetTempPath(),"hopbox-setup-"+Guid.NewGuid().ToString("N")+".txt");
                    try {
                        var start=new ProcessStartInfo(Assembly.GetExecutingAssembly().Location,"--quiet --no-launch --install-dir \""+options.Target+"\" --components "+"core"+" --progress-file \""+progressFile+"\"") { UseShellExecute=true,Verb="runas" };
                        using(var child=Process.Start(start)) {
                            while(!child.WaitForExit(250)) if(File.Exists(progressFile)) try { string[] lines=File.ReadAllLines(progressFile); int value; if(lines.Length>1&&Int32.TryParse(lines[0],out value)) UpdateProgress(value,lines[1]); } catch(IOException) {}
                            result=child.ExitCode;
                        }
                    } finally { if(File.Exists(progressFile)) { if(result!=0) UpdateProgress(-1,File.ReadAllText(progressFile)); File.Delete(progressFile); } }
                } else result=Setup.Install(options,UpdateProgress);
            } catch(Exception error) { UpdateProgress(-1,error.Message); }
            BeginInvoke((Action)delegate {
                busy=false; Result=result;
                if(result==0) {
                    finished=true; progress.Value=100; status.Text="Installed in:\n"+options.Target;
                    try { Setup.Shortcuts(options.Target,desktop.Checked); } catch(Exception error) { status.Text="Installed. Shortcut creation failed: "+error.Message; }
                    // The original user process opens the app, never the elevated child.
                    if(open.Checked) try { Setup.Launch(options.Target); } catch(Exception error) { status.Text="Installed. Open HopBox-Infer.exe manually: "+error.Message; }
                    install.Text="Finish"; install.Enabled=true;
                } else { destination.Enabled=browse.Enabled=desktop.Enabled=open.Enabled=install.Enabled=cancel.Enabled=true; }
            });
        }) { IsBackground=true }.Start();
    }
}
