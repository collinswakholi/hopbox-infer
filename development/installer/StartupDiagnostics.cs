using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Web.Script.Serialization;

public sealed class PackageFile {
    public string path, sha256;
    public long size;
    public bool required, fast_hash;
}
public sealed class PackageFiles { public string version; public PackageFile[] files; }

public static class StartupDiagnostics {
    public static readonly string Root = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory).TrimEnd('\\');
    public static string StateDirectory() {
        string configured=Environment.GetEnvironmentVariable("HOPBOX_STATE_ROOT");
        string path=String.IsNullOrWhiteSpace(configured)?Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"HopBox-Infer"):Path.GetFullPath(configured);
        Directory.CreateDirectory(path); return path;
    }
    public static string NewLog() {
        string folder=StateDirectory();
        // Keep recent startup attempts without deleting any measurement outputs.
        var logs=new List<string>(Directory.GetFiles(folder,"startup-*.log"));
        logs.Sort(StringComparer.Ordinal);
        for(int i=0;i<logs.Count-9;i++) try { File.Delete(logs[i]); } catch(IOException) {} catch(UnauthorizedAccessException) {}
        return Path.Combine(folder,"startup-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss")+"-"+Guid.NewGuid().ToString("N").Substring(0,8)+".log");
    }
    public static void Append(string path,string message) {
        try { using(var file=new FileStream(path,FileMode.Append,FileAccess.Write,FileShare.ReadWrite)) using(var writer=new StreamWriter(file)) writer.WriteLine(DateTime.UtcNow.ToString("o")+" "+message); } catch {}
    }
    public static string Hash(string path) {
        using(var stream=File.OpenRead(path)) using(var hash=SHA256.Create()) return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-","").ToLowerInvariant();
    }
    public static List<string> Check(bool full) {
        var errors=new List<string>();
        string manifest=Path.Combine(Root,"package-files.json");
        if(!File.Exists(manifest)) { errors.Add("Missing package-files.json. Install the complete updated package; copying only the EXE is insufficient."); return errors; }
        var data=new JavaScriptSerializer().Deserialize<PackageFiles>(File.ReadAllText(manifest));
        if(data==null||data.files==null||data.files.Length==0) throw new IOException("The package integrity inventory is invalid.");
        foreach(var item in data.files) {
            if(!full&&!item.required) continue;
            string path=Path.GetFullPath(Path.Combine(Root,item.path.Replace('/',Path.DirectorySeparatorChar)));
            if(!path.StartsWith(Root+"\\",StringComparison.OrdinalIgnoreCase)) throw new IOException("Unsafe path in package inventory.");
            if(!File.Exists(path)) { errors.Add("Missing file: "+item.path); continue; }
            try {
                if(new FileInfo(path).Length!=item.size) errors.Add("Changed file size: "+item.path);
                else if((full||item.fast_hash)&&Hash(path)!=item.sha256) errors.Add("Checksum mismatch: "+item.path);
            } catch(Exception error) { errors.Add("Cannot read "+item.path+": "+error.Message); }
        }
        return errors;
    }
    public static string Tail(string path) {
        try { using(var file=new FileStream(path,FileMode.Open,FileAccess.Read,FileShare.ReadWrite)) {
            if(file.Length>12000) file.Seek(-12000,SeekOrigin.End);
            using(var reader=new StreamReader(file)) return reader.ReadToEnd();
        } } catch { return "No Python diagnostic output was written. The runtime may have been prevented from starting."; }
    }
    public static string ExitReason(int code) {
        uint value=unchecked((uint)code);
        string reason="The runtime exited before model initialization finished.";
        if(value==0xc000001d) reason="A native dependency attempted an unsupported CPU instruction.";
        if(value==0xc0000135) reason="Windows could not load a required DLL.";
        if(value==0xc000007b) reason="Windows reported an invalid executable/DLL format.";
        if(value==0xc0000005) reason="A native dependency crashed with an access violation; inspect the diagnostic log and Windows Application event.";
        return reason+" Exit code "+code+" (0x"+value.ToString("X8")+").";
    }
    public static string Diagnose(string output) {
        string path=String.IsNullOrWhiteSpace(output)?Path.Combine(StateDirectory(),"diagnostics-"+DateTime.UtcNow.ToString("yyyyMMdd-HHmmss")+".json"):Path.GetFullPath(output);
        var errors=Check(true);
        var data=new Dictionary<string,object> {
            {"version",Branding.Version},{"install_root",Root},{"windows",Environment.OSVersion.ToString()},
            {"os_64bit",Environment.Is64BitOperatingSystem},{"processor",Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER")},
            {"launcher_sha256",Hash(System.Reflection.Assembly.GetExecutingAssembly().Location)},
            {"integrity_pass",errors.Count==0},{"integrity_errors",errors},{"generated_utc",DateTime.UtcNow.ToString("o")},
            {"note","Local integrity report only; hashes do not establish publisher trust. No security policy was changed and nothing was uploaded."}
        };
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path,new JavaScriptSerializer().Serialize(data)); return path;
    }
}
