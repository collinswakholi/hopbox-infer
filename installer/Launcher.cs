using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;

public sealed class Splash : Form {
    readonly Label status;
    readonly string address, startupId = Guid.NewGuid().ToString("N");
    Process server;
    bool completed;
    volatile bool closing;
    readonly Stopwatch elapsed = Stopwatch.StartNew();
    readonly string diagnosticLog = StartupDiagnostics.NewLog();
    public int Result = 2;

    public Splash(string address) {
        this.address = address;
        Text = Branding.Name;
        Icon = Icon.ExtractAssociatedIcon(Assembly.GetExecutingAssembly().Location);
        StartPosition = FormStartPosition.CenterScreen;
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = false; MinimizeBox = false;
        AutoScaleMode = AutoScaleMode.Dpi;
        ClientSize = new Size(540, 324);
        BackColor = Color.FromArgb(242, 247, 245);
        ForeColor = Color.FromArgb(23, 76, 60);
        Font = new Font("Segoe UI", 10);
        var picture = new PictureBox { Bounds = new Rectangle(28, 28, 138, 138), SizeMode = PictureBoxSizeMode.Zoom };
        using (var resource = Assembly.GetExecutingAssembly().GetManifestResourceStream("HopBoxLogo"))
        using (var source = Image.FromStream(resource)) picture.Image = new Bitmap(source);
        Controls.Add(picture);
        AddLabel(Branding.Name, 188, 35, 325, 50, 20, FontStyle.Bold);
        AddLabel(Branding.Author, 190, 94, 310, 26, 12, FontStyle.Regular);
        var link = new LinkLabel { Text = Branding.Github.Replace("https://", ""), Bounds = new Rectangle(190, 128, 320, 25), LinkColor = Color.FromArgb(44,120,124), ActiveLinkColor = Color.FromArgb(23,76,60) };
        link.LinkClicked += delegate { try { Process.Start(new ProcessStartInfo(Branding.Github) { UseShellExecute = true }); } catch { } };
        Controls.Add(link);
        AddLabel("Version " + Branding.Version, 32, 184, 475, 23, 10, FontStyle.Regular);
        var date = DateTime.ParseExact(Branding.Developed, "yyyy-MM-dd", CultureInfo.InvariantCulture);
        AddLabel("Developed: " + date.ToString("MMMM d, yyyy", CultureInfo.GetCultureInfo("en-US")), 32, 210, 475, 23, 10, FontStyle.Regular);
        status = AddLabel("Starting HopBox Infer…", 32, 256, 475, 25, 10, FontStyle.Regular);
        Controls.Add(new ProgressBar { Bounds = new Rectangle(32, 289, 475, 5), Style = ProgressBarStyle.Marquee, MarqueeAnimationSpeed = 32 });
        FormClosed += delegate { Trace("splash-closed"); };
    }
    Label AddLabel(string text, int x, int y, int width, int height, float size, FontStyle weight) {
        var label = new Label { Text = text, Bounds = new Rectangle(x,y,width,height), Font = new Font("Segoe UI",size,weight) };
        Controls.Add(label); return label;
    }
    void Trace(string name) {
        StartupDiagnostics.Append(diagnosticLog,name);
        string file = Environment.GetEnvironmentVariable("HOPBOX_STARTUP_TRACE");
        if (String.IsNullOrEmpty(file)) return;
        try { File.AppendAllText(file, name + " " + elapsed.Elapsed.TotalSeconds.ToString("F3",CultureInfo.InvariantCulture) + Environment.NewLine); } catch { }
    }
    Dictionary<string,object> Ready() {
        var query = (HttpWebRequest)WebRequest.Create(address + "/api/ready");
        query.Proxy = null; query.Timeout = 800;
        using (var response = query.GetResponse())
        using (var text = new StreamReader(response.GetResponseStream()))
            return new JavaScriptSerializer().Deserialize<Dictionary<string,object>>(text.ReadToEnd());
    }
    void StopServer() {
        if (server == null) return;
        try {
            var ready = Ready();
            if ((string)ready["startup_id"] == startupId) {
                var call = (HttpWebRequest)WebRequest.Create(address + "/api/exit");
                call.Proxy = null; call.Method = "POST"; call.ContentType = "application/json"; call.Timeout = 1500;
                using (var body = new StreamWriter(call.GetRequestStream())) body.Write("{}");
                using (call.GetResponse()) { }
                if (server.WaitForExit(30000)) return;
            }
        } catch { }
        try { if (!server.HasExited) server.Kill(); } catch { }
    }
    public void StartServer() {
        Trace("splash-shown");
        string core = StartupDiagnostics.Root;
        Trace("Install directory: "+core);
        var integrity=StartupDiagnostics.Check(false);
        if(integrity.Count!=0) throw new IOException("The installation is incomplete or changed.\n"+String.Join("\n",integrity.GetRange(0,Math.Min(8,integrity.Count)).ToArray())+"\nAn existing app folder does not establish that its files are intact. Have IT review quarantine events, then repair from the approved complete installer.");
        var probe=new TcpListener(IPAddress.Loopback,new Uri(address).Port);
        try { probe.Start(); } catch(SocketException) { throw new IOException("The local port "+new Uri(address).Port+" is already in use. Close the other HopBox window or set HOPBOX_PORT to a free port."); } finally { probe.Stop(); }
        string interpreter = Path.Combine(core,"python","pythonw.exe");
        var start = new ProcessStartInfo(interpreter,"-B -m hopbox_bootstrap");
        start.WorkingDirectory = core; start.UseShellExecute = false; start.CreateNoWindow = true;
        start.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
        start.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"] = "1";
        start.EnvironmentVariables["HOPBOX_PACKAGED"] = "1";
        start.EnvironmentVariables["HOPBOX_APP_ROOT"] = Path.Combine(core,"app");
        start.EnvironmentVariables["HOPBOX_STARTUP_ID"] = startupId;
        start.EnvironmentVariables["HOPBOX_STARTUP_LOG"] = diagnosticLog;
        start.EnvironmentVariables.Remove("PYTHONPATH");
        start.EnvironmentVariables["PATH"] = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows),"System32");
        server = Process.Start(start);
        Trace("server-started");
        new Thread(delegate() {
            try {
                bool retriedOnCpu = false;
                DateTime deadline = DateTime.UtcNow.AddMinutes(5);
                while (!closing && DateTime.UtcNow < deadline) {
                    if (server.HasExited) {
                        int exitCode = server.ExitCode;
                        // A native GPU-driver crash cannot be caught by Python.
                        // Extend the existing Python GPU-to-CPU fallback across
                        // that process boundary, once, during startup only.
                        if (!retriedOnCpu && unchecked((uint)exitCode) == 0xc0000005 &&
                            !String.Equals(start.EnvironmentVariables["HOPBOX_DEVICE"],"cpu",StringComparison.OrdinalIgnoreCase)) {
                            Trace("native-startup-exit " + StartupDiagnostics.ExitReason(exitCode));
                            if (closing) return;
                            retriedOnCpu = true;
                            start.EnvironmentVariables["HOPBOX_DEVICE"] = "cpu";
                            server.Dispose();
                            server = Process.Start(start);
                            Trace("cpu-startup-recovery");
                            if (closing) { StopServer(); return; }
                            deadline = DateTime.UtcNow.AddMinutes(5);
                            BeginInvoke((Action)delegate { if (!closing) status.Text = "GPU startup stopped. Opening on CPU…"; });
                            Thread.Sleep(250);
                            continue;
                        }
                        throw new IOException(StartupDiagnostics.ExitReason(exitCode));
                    }
                    Dictionary<string,object> reply = null;
                    try { reply = Ready(); } catch (WebException) { } catch (ArgumentException) { }
                    if (reply != null && reply.ContainsKey("startup_id") && (string)reply["startup_id"] == startupId) {
                        if (reply.ContainsKey("failed") && (bool)reply["failed"]) throw new IOException((string)reply["text"]);
                        if ((bool)reply["ok"]) {
                            Trace("models-ready");
                            BeginInvoke((Action)delegate { Result=0; completed = true; Close(); });
                            return;
                        }
                        BeginInvoke((Action)delegate { if (!closing) status.Text = retriedOnCpu ? "Loading models on CPU…" : "Loading models…"; });
                    }
                    Thread.Sleep(250);
                }
                if (!closing) throw new IOException("Startup took longer than five minutes. Close other applications and try again.");
            } catch (Exception error) {
                Trace("startup-error");
                StopServer();
                if (!closing) BeginInvoke((Action)delegate { ShowFailure(error); });
            }
        }) { IsBackground = true }.Start();
    }
    public void ShowFailure(Exception error) {
        Result=1; completed=true;
        StartupDiagnostics.Append(diagnosticLog,error.ToString());
        string tail=StartupDiagnostics.Tail(diagnosticLog);
        // Keep the dialog readable; the full traceback/native stack is on disk.
        if(tail.Length>1200) tail=tail.Substring(tail.Length-1200);
        MessageBox.Show(this,error.Message+"\n\nDiagnostic log:\n"+diagnosticLog+"\n\n"+tail,Branding.Name+" startup",MessageBoxButtons.OK,MessageBoxIcon.Error);
        Close();
    }
    protected override void OnFormClosing(FormClosingEventArgs e) {
        closing = true;
        if (!completed) {
            Trace("startup-cancelled");
            new Thread(StopServer).Start();
        }
        base.OnFormClosing(e);
    }
    public void Render(string file) {
        Show(); Application.DoEvents();
        using (var bitmap = new Bitmap(Width, Height)) {
            DrawToBitmap(bitmap,new Rectangle(Point.Empty,Size)); bitmap.Save(file);
        }
    }
}
public static class Launcher {
    [STAThread] public static int Main(string[] args) {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        try {
            if(args.Length>=1&&args[0]=="--diagnose") {
                if(args.Length>2) throw new ArgumentException("Usage: HopBox-Infer.exe --diagnose [report.json]");
                string report=StartupDiagnostics.Diagnose(args.Length==2?args[1]:null);
                if(args.Length==1) MessageBox.Show("Local integrity report saved to:\n"+report+"\nNo settings were changed and no files were uploaded.",Branding.Name);
                var result=new JavaScriptSerializer().Deserialize<Dictionary<string,object>>(File.ReadAllText(report));
                return (bool)result["integrity_pass"]?0:1;
            }
            if(args.Length==1&&args[0]=="--cpu") Environment.SetEnvironmentVariable("HOPBOX_DEVICE","cpu");
            else if(args.Length==1&&args[0]=="--gpu-generic") Environment.SetEnvironmentVariable("HOPBOX_DML_GENERIC","1");
            else if(args.Length!=0&&!(args.Length==2&&args[0]=="--render-splash")) throw new ArgumentException("Options: --cpu, --gpu-generic, --diagnose [report.json], --render-splash file.png");
            string port = Environment.GetEnvironmentVariable("HOPBOX_PORT") ?? "8765";
            int number; if (!Int32.TryParse(port,out number) || number < 1024 || number > 65535) throw new ArgumentException("Invalid HopBox Infer port.");
            using (var splash = new Splash("http://127.0.0.1:" + number)) {
                if (args.Length == 2 && args[0] == "--render-splash") { splash.Render(args[1]); return 0; }
                splash.Shown += delegate { try { splash.StartServer(); } catch (Exception error) { splash.ShowFailure(error); } };
                Application.Run(splash);
                return splash.Result;
            }
        } catch(Exception error) { MessageBox.Show(error.Message,Branding.Name,MessageBoxButtons.OK,MessageBoxIcon.Error); return 1; }
    }
}
