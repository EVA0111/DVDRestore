// DVD Restore.exe - starts the app from wherever the folder is (portable: no install, no registry, no PATH).
// Built by tools\build_portable.py with the C# compiler that comes with Windows (.NET Framework 4).
//   <folder>\DVD Restore.exe      this launcher
//   <folder>\runtime\electron\    the window (shipped)          <folder>\app\  core\  models\
//   <folder>\runtime\python\ ...  installed by the Dependencies window on first start
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyTitle("DVD Restore")]
[assembly: System.Reflection.AssemblyProduct("DVD Restore")]
[assembly: System.Reflection.AssemblyDescription("DVD Restore v0.1 alpha")]
[assembly: System.Reflection.AssemblyVersion("0.1.0.0")]
[assembly: System.Reflection.AssemblyFileVersion("0.1.0.0")]

static class Launcher
{
    static void Fail(string text)
    {
        MessageBox.Show(text, "DVD Restore", MessageBoxButtons.OK, MessageBoxIcon.Warning);
    }

    static string Quote(string a)
    {
        if (a.Length > 0 && a.IndexOfAny(new[] { ' ', '\t', '"' }) < 0) return a;
        var sb = new StringBuilder("\"");
        int slashes = 0;
        foreach (char ch in a)
        {
            if (ch == '\\') { slashes++; continue; }
            if (ch == '"') sb.Append('\\', slashes * 2 + 1);
            else sb.Append('\\', slashes);
            slashes = 0;
            sb.Append(ch);
        }
        sb.Append('\\', slashes * 2);
        return sb.Append('"').ToString();
    }

    [STAThread]
    static int Main(string[] args)
    {
        string root = Path.GetDirectoryName(Path.GetFullPath(Application.ExecutablePath));
        string temp = Path.GetFullPath(Path.GetTempPath()).TrimEnd('\\');
        if (root.StartsWith(temp, StringComparison.OrdinalIgnoreCase) && root.IndexOf(".zip", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            Fail("DVD Restore is running from inside the zip file.\n\nUnpack the whole zip first (right-click > Extract All), then start DVD Restore.exe in the unpacked folder.");
            return 1;
        }
        string electron = Path.Combine(root, "runtime", "electron", "electron.exe");
        string app = Path.Combine(root, "app");
        if (!File.Exists(electron) || !File.Exists(Path.Combine(app, "main.js")))
        {
            Fail("Parts of DVD Restore are missing next to this file:\n\n" +
                 (File.Exists(electron) ? "" : "  runtime\\electron\\electron.exe\n") +
                 (File.Exists(Path.Combine(app, "main.js")) ? "" : "  app\\main.js\n") +
                 "\nUnpack the whole download again, keeping all folders together.");
            return 1;
        }
        var cmd = new StringBuilder(Quote(app));
        foreach (string a in args) cmd.Append(' ').Append(Quote(a));
        var psi = new ProcessStartInfo(electron, cmd.ToString());
        psi.UseShellExecute = false;
        psi.WorkingDirectory = root;
        try { Process.Start(psi); }
        catch (Exception e) { Fail("Could not start DVD Restore:\n\n" + e.Message); return 1; }
        return 0;
    }
}
