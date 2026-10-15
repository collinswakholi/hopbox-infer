using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace HopBoxNative {
    public static class Dialogs {
        public static string PickImport(string title, string start) {
            return ShellDialog.Show(title, start, false);
        }
        public static string PickFolder(string title, string start) {
            return ShellDialog.Show(title, start, true);
        }
    }

    static class ShellDialog {
        const uint BM_CLICK = 0x00F5;
        const uint WM_LBUTTONUP = 0x0202;
        const uint WM_COMMAND = 0x0111;
        const uint WM_APP_ACCEPT = 0x8000 + 31;
        const uint SWP_NOSIZE = 0x0001;
        const uint SWP_NOMOVE = 0x0002;
        const uint SWP_SHOWWINDOW = 0x0040;
        const int GWLP_WNDPROC = -4;

        enum SelKind { Failed, Empty, Found }

        static IFileOpenDialog _dialog;
        static bool _foldersOnly;
        static bool _done;
        static List<string> _captured = new List<string>();
        static IntPtr _dialogHwnd;
        static IntPtr _buttonHwnd;
        static bool _dialogComctl;
        static bool _buttonComctl;
        static bool _dialogClassic;
        static bool _buttonClassic;
        static IntPtr _dialogOld;
        static IntPtr _buttonOld;
        static bool _foregrounded;
        static IntPtr _keyHook;
        static SubclassProc _dialogProc;
        static SubclassProc _buttonProc;
        static WndProc _dialogClassicProc;
        static WndProc _buttonClassicProc;
        static IntPtr _actCtx = new IntPtr(-1);
        static bool _comctlReady;

        public static void SetDpiAware() {
            try { SetProcessDpiAwarenessContext(new IntPtr(-4)); } catch { }
            try { SetProcessDpiAwareness(2); } catch { }
            try { SetThreadDpiAwarenessContext(new IntPtr(-4)); } catch { }
        }

        public static string Show(string title, string start, bool foldersOnly) {
            Reset();
            ResetLog();
            _foldersOnly = foldersOnly;
            SetDpiAware();
            Application.EnableVisualStyles();
            ActivateComctl();
            IFileOpenDialog dialog = (IFileOpenDialog)new FileOpenDialogRCW();
            _dialog = dialog;
            FOS options = FOS.PICKFOLDERS | FOS.FORCEFILESYSTEM | FOS.PATHMUSTEXIST;
            if (!foldersOnly) options |= FOS.ALLOWMULTISELECT;
            dialog.SetOptions(options);
            string shownTitle = string.IsNullOrEmpty(title) ? "Import" : title;
            dialog.SetTitle(shownTitle);
            dialog.SetOkButtonLabel(foldersOnly ? "Select folder" : "Import");
            if (!string.IsNullOrEmpty(start)) {
                try {
                    Guid iid = typeof(IShellItem).GUID;
                    IShellItem item;
                    SHCreateItemFromParsingName(start, IntPtr.Zero, ref iid, out item);
                    dialog.SetFolder(item);
                } catch (Exception ex) {
                    Debug("start " + ex.Message);
                }
            }
            Form owner = CreateOwner();
            System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
            timer.Interval = 40;
            timer.Tick += delegate {
                IntPtr hwnd = FindTopWindow(shownTitle);
                if (hwnd == IntPtr.Zero) return;
                timer.Stop();
                ForceForeground(hwnd);
                CenterDialog(hwnd);
            };
            int hr = 1;
            try {
                owner.Show();
                ForceForeground(owner.Handle);
                timer.Start();
                hr = dialog.Show(owner.Handle);
            } finally {
                timer.Stop();
                timer.Dispose();
                try { owner.Dispose(); } catch { }
            }
            if (hr != 0) return "";
            List<string> results = new List<string>();
            try { CollectResults(dialog, results); } catch (Exception ex) { Debug("results " + ex.Message); }
            if (results.Count == 0) {
                try {
                    IShellItem one;
                    dialog.GetResult(out one);
                    string path;
                    one.GetDisplayName(SIGDN.FILESYSPATH, out path);
                    if (!string.IsNullOrEmpty(path)) results.Add(path);
                } catch { }
            }
            if (results.Count == 0) {
                string current = CurrentFolder(dialog);
                if (!string.IsNullOrEmpty(current)) results.Add(current);
            }
            return string.Join("\n", results.ToArray());
        }

        static IntPtr FindTopWindow(string title) {
            WindowSearch search = new WindowSearch();
            search.Title = title;
            EnumWindowsEntry(search.Visit, IntPtr.Zero);
            return search.Found;
        }

        class WindowSearch {
            public string Title;
            public IntPtr Found;
            public bool Visit(IntPtr hwnd, IntPtr lparam) {
                if (!IsWindowVisible(hwnd)) return true;
                StringBuilder text = new StringBuilder(512);
                GetWindowText(hwnd, text, text.Capacity);
                if (text.ToString() == Title) {
                    Found = hwnd;
                    return false;
                }
                return true;
            }
        }

        static void Reset() {
            _dialog = null;
            _done = false;
            _captured = new List<string>();
            _dialogHwnd = IntPtr.Zero;
            _buttonHwnd = IntPtr.Zero;
            _dialogComctl = false;
            _buttonComctl = false;
            _dialogClassic = false;
            _buttonClassic = false;
            _dialogOld = IntPtr.Zero;
            _buttonOld = IntPtr.Zero;
            _foregrounded = false;
            _keyHook = IntPtr.Zero;
        }

        static void OnTick(object sender, EventArgs e) {
            try {
                if (_dialog != null) Install(_dialog);
            } catch (Exception ex) {
                Debug("tick " + ex.Message);
            }
        }

        static void Install(IFileOpenDialog dialog) {
            if (dialog == null || _foldersOnly) return;
            _dialog = dialog;
            if (_dialogHwnd == IntPtr.Zero || !IsWindow(_dialogHwnd)) {
                try {
                    IOleWindow ole = (IOleWindow)dialog;
                    IntPtr hwnd;
                    ole.GetWindow(out hwnd);
                    _dialogHwnd = hwnd;
                } catch (Exception ex) {
                    Debug("hwnd " + ex.Message);
                    return;
                }
            }
            if (_dialogHwnd == IntPtr.Zero || !IsWindow(_dialogHwnd)) return;
            ActivateComctl();
            if (!_dialogComctl && !_dialogClassic) {
                _dialogProc = DialogProc;
                _dialogClassicProc = DialogClassic;
                if (TrySubclass(_dialogHwnd, _dialogProc, new UIntPtr(0x484F5001))) {
                    _dialogComctl = true;
                    Debug("dialog subclass comctl");
                } else if (TryClassic(_dialogHwnd, _dialogClassicProc, out _dialogOld)) {
                    _dialogClassic = true;
                    Debug("dialog subclass classic");
                } else {
                    Debug("dialog subclass failed");
                }
            }
            IntPtr ok = GetDlgItem(_dialogHwnd, 1);
            if (ok != IntPtr.Zero && ok != _buttonHwnd) {
                _buttonHwnd = ok;
                _buttonProc = ButtonProc;
                _buttonClassicProc = ButtonClassic;
                if (TrySubclass(ok, _buttonProc, new UIntPtr(0x484F5002))) {
                    _buttonComctl = true;
                    Debug("button subclass comctl hwnd=" + ok.ToInt64() + " dialog=" + _dialogHwnd.ToInt64());
                } else if (TryClassic(ok, _buttonClassicProc, out _buttonOld)) {
                    _buttonClassic = true;
                    Debug("button subclass classic");
                } else {
                    Debug("button subclass failed");
                }
            }
            if (!_foregrounded) {
                _foregrounded = true;
                ForceForeground(_dialogHwnd);
                CenterDialog(_dialogHwnd);
                RECT placed;
                if (GetWindowRect(_dialogHwnd, out placed)) {
                    Debug("rect " + placed.Left + "," + placed.Top + " " + (placed.Right - placed.Left) + "x" + (placed.Bottom - placed.Top));
                }
            }
        }

        static void RemoveHooks() {
            if (_keyHook != IntPtr.Zero) {
                UnhookWindowsHookEx(_keyHook);
                _keyHook = IntPtr.Zero;
            }
            if (_buttonHwnd != IntPtr.Zero && IsWindow(_buttonHwnd)) {
                if (_buttonComctl && _buttonProc != null) {
                    try { RemoveWindowSubclass(_buttonHwnd, _buttonProc, new UIntPtr(0x484F5002)); } catch { }
                } else if (_buttonClassic && _buttonOld != IntPtr.Zero) {
                    ReplaceProc(_buttonHwnd, _buttonOld);
                }
            }
            if (_dialogHwnd != IntPtr.Zero && IsWindow(_dialogHwnd)) {
                if (_dialogComctl && _dialogProc != null) {
                    try { RemoveWindowSubclass(_dialogHwnd, _dialogProc, new UIntPtr(0x484F5001)); } catch { }
                } else if (_dialogClassic && _dialogOld != IntPtr.Zero) {
                    ReplaceProc(_dialogHwnd, _dialogOld);
                }
            }
        }

        static bool OnButtonMessage(IntPtr hWnd, uint msg, IntPtr lParam) {
            if (msg == BM_CLICK || msg == WM_LBUTTONUP || msg == 0x0201) {
                IntPtr focus = GetFocus();
                Debug("btn msg=" + msg + " focus=" + ClassNameOf(focus) + " id=" + GetDlgCtrlID(focus) + " inside=" + (msg != WM_LBUTTONUP || Inside(hWnd, lParam)));
            }
            if (msg != BM_CLICK && msg != WM_LBUTTONUP) return false;
            if (_done) return true;
            if (msg == WM_LBUTTONUP && !Inside(hWnd, lParam)) return false;
            return TakeButton(true);
        }

        static IntPtr ButtonProc(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam, UIntPtr id, UIntPtr data) {
            try {
                if (OnButtonMessage(hWnd, msg, lParam)) return IntPtr.Zero;
            } catch (Exception ex) {
                Debug("button " + ex.Message);
            }
            return DefSubclassProc(hWnd, msg, wParam, lParam);
        }

        static IntPtr ButtonClassic(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam) {
            try {
                if (OnButtonMessage(hWnd, msg, lParam)) return IntPtr.Zero;
            } catch (Exception ex) {
                Debug("button classic " + ex.Message);
            }
            return CallWindowProc(_buttonOld, hWnd, msg, wParam, lParam);
        }

        static IntPtr DialogProc(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam, UIntPtr id, UIntPtr data) {
            try {
                if (OnDialogMessage(msg, wParam)) return IntPtr.Zero;
            } catch (Exception ex) {
                Debug("dialog " + ex.Message);
            }
            return DefSubclassProc(hWnd, msg, wParam, lParam);
        }

        static IntPtr DialogClassic(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam) {
            try {
                if (OnDialogMessage(msg, wParam)) return IntPtr.Zero;
            } catch (Exception ex) {
                Debug("dialog classic " + ex.Message);
            }
            return CallWindowProc(_dialogOld, hWnd, msg, wParam, lParam);
        }

        static bool OnDialogMessage(uint msg, IntPtr wParam) {
            if (msg == WM_APP_ACCEPT) {
                Debug("accept message");
                if (!_done) return true;
                try {
                    if (_dialog != null) _dialog.Close(0);
                    Debug("accept close returned");
                } catch (Exception ex) {
                    Debug("accept close " + ex.Message);
                }
                return true;
            }
            if (msg == WM_COMMAND) {
                int packed = wParam.ToInt32();
                int id = packed & 0xFFFF;
                int code = (packed >> 16) & 0xFFFF;
                Debug("cmd id=" + id + " code=" + code);
                if (id == 1 && code == 0) return TakeButton(true);
            }
            return false;
        }

        static IntPtr KeyProc(int nCode, IntPtr wParam, IntPtr lParam) {
            try {
                if (nCode >= 0 && !_done && !_foldersOnly && wParam.ToInt32() == 0x0D) {
                    long lp = lParam.ToInt64() & 0xFFFFFFFFL;
                    bool up = (lp & 0x80000000L) != 0;
                    bool prior = (lp & 0x40000000L) != 0;
                    bool alt = (lp & 0x20000000L) != 0;
                    if (!up && !prior && !alt && !FocusIsTyping()) {
                        List<string> paths = new List<string>();
                        bool hadFolder;
                        bool take = Decide(out paths, out hadFolder, false);
                        if (take && hadFolder) {
                            Commit(paths);
                            return (IntPtr)1;
                        }
                    }
                }
            } catch (Exception ex) {
                Debug("key " + ex.Message);
            }
            return CallNextHookEx(_keyHook, nCode, wParam, lParam);
        }

        static bool TakeButton(bool ignoreFocus) {
            if (_done) return true;
            List<string> paths = new List<string>();
            bool hadFolder;
            bool take = Decide(out paths, out hadFolder, ignoreFocus);
            if (!take) return false;
            Commit(paths);
            return true;
        }

        static void Commit(List<string> paths) {
            if (_done) return;
            _done = true;
            _captured = paths;
            Debug("commit " + paths.Count + " " + (paths.Count > 0 ? paths[0] : ""));
            ReleaseCapture();
            try {
                if (_dialog != null) _dialog.Close(0);
                Debug("close returned");
            } catch (Exception ex) {
                Debug("close ex " + ex.GetType().Name + " " + ex.Message);
            }
            PostQuitMessage(0);
            Debug("quit posted");
        }

        static bool Decide(out List<string> paths, out bool hadFolder, bool ignoreFocus) {
            paths = new List<string>();
            hadFolder = false;
            if (_dialog == null) return false;
            if (!ignoreFocus && FocusIsTyping()) return false;
            List<string> viewed = new List<string>();
            SelKind view = TryView(viewed);
            List<string> chosen = null;
            if (view == SelKind.Found) chosen = viewed;
            else if (view == SelKind.Empty) chosen = new List<string>();
            else {
                List<string> selected = new List<string>();
                SelKind dialog = TrySelected(selected);
                if (dialog == SelKind.Found) chosen = selected;
                else if (dialog == SelKind.Empty) chosen = new List<string>();
                else {
                    List<string> named = new List<string>();
                    SelKind fromName = TryName(named);
                    if (fromName == SelKind.Found) chosen = named;
                    else if (fromName == SelKind.Empty) chosen = new List<string>();
                    else return false;
                }
            }
            Debug("decide view=" + view + " count=" + (chosen == null ? -1 : chosen.Count));
            if (chosen == null || chosen.Count == 0) {
                string current = CurrentFolder(_dialog);
                if (string.IsNullOrEmpty(current)) return false;
                paths.Add(current);
                hadFolder = false;
                return true;
            }
            if (!AnyDir(chosen)) return false;
            paths.AddRange(chosen);
            hadFolder = true;
            return true;
        }

        static SelKind TryView(List<string> paths) {
            IntPtr browserPtr = IntPtr.Zero;
            try {
                IShellService service = (IShellService)_dialog;
                Guid sid = new Guid("4C96BE40-915C-11CF-99D3-00AA004AE837");
                Guid iidBrowser = new Guid("000214E2-0000-0000-C000-000000000046");
                int hr = service.QueryService(ref sid, ref iidBrowser, out browserPtr);
                if (hr != 0 || browserPtr == IntPtr.Zero) return SelKind.Failed;
                IShellBrowser browser = (IShellBrowser)Marshal.GetObjectForIUnknown(browserPtr);
                IShellView view;
                browser.QueryActiveShellView(out view);
                IFolderView folderView = (IFolderView)view;
                int count;
                folderView.ItemCount(1, out count);
                if (count <= 0) return SelKind.Empty;
                Guid iidItems = typeof(IShellItemArray).GUID;
                IntPtr arrayPtr;
                int ihr = folderView.Items(1, ref iidItems, out arrayPtr);
                if (ihr != 0 || arrayPtr == IntPtr.Zero) return SelKind.Failed;
                try {
                    IShellItemArray items = (IShellItemArray)Marshal.GetObjectForIUnknown(arrayPtr);
                    ReadItems(items, paths);
                } finally {
                    Marshal.Release(arrayPtr);
                }
                if (paths.Count == 0) return SelKind.Failed;
                return SelKind.Found;
            } catch (Exception ex) {
                Debug("view " + ex.Message);
                return SelKind.Failed;
            } finally {
                if (browserPtr != IntPtr.Zero) Marshal.Release(browserPtr);
            }
        }

        static SelKind TrySelected(List<string> paths) {
            try {
                IShellItemArray items;
                _dialog.GetSelectedItems(out items);
                uint count;
                items.GetCount(out count);
                if (count == 0) return SelKind.Empty;
                ReadItems(items, paths);
                if (paths.Count == 0) return SelKind.Failed;
                return SelKind.Found;
            } catch (Exception ex) {
                Debug("selected " + ex.Message);
                return SelKind.Failed;
            }
        }

        static SelKind TryName(List<string> paths) {
            try {
                string text;
                _dialog.GetFileName(out text);
                if (string.IsNullOrEmpty(text) || text.Trim().Length == 0) return SelKind.Empty;
                string folder = CurrentFolder(_dialog);
                List<string> tokens = SplitNames(text);
                for (int i = 0; i < tokens.Count; i++) {
                    string token = tokens[i];
                    if (token.IndexOf('*') >= 0 || token.IndexOf('?') >= 0) continue;
                    string path = token;
                    if (!Path.IsPathRooted(token) && !string.IsNullOrEmpty(folder)) path = Path.Combine(folder, token);
                    if (Directory.Exists(path) || File.Exists(path)) paths.Add(path);
                }
                if (paths.Count == 0) return SelKind.Empty;
                return SelKind.Found;
            } catch (Exception ex) {
                Debug("name " + ex.Message);
                return SelKind.Failed;
            }
        }

        static List<string> SplitNames(string text) {
            List<string> tokens = new List<string>();
            text = text.Trim();
            if (text.IndexOf('"') < 0) {
                tokens.Add(text);
                return tokens;
            }
            int i = 0;
            while (i < text.Length) {
                if (text[i] != '"') {
                    i++;
                    continue;
                }
                int j = text.IndexOf('"', i + 1);
                if (j < 0) break;
                tokens.Add(text.Substring(i + 1, j - i - 1));
                i = j + 1;
            }
            return tokens;
        }

        static void ReadItems(IShellItemArray items, List<string> paths) {
            uint count;
            items.GetCount(out count);
            for (uint i = 0; i < count; i++) {
                IShellItem item;
                items.GetItemAt(i, out item);
                string path;
                item.GetDisplayName(SIGDN.FILESYSPATH, out path);
                if (!string.IsNullOrEmpty(path)) paths.Add(path);
            }
        }

        static bool AnyDir(List<string> paths) {
            for (int i = 0; i < paths.Count; i++) {
                if (IsDirectory(paths[i])) return true;
            }
            return false;
        }

        static bool IsDirectory(string path) {
            try {
                FileAttributes attr = File.GetAttributes(path);
                return (attr & FileAttributes.Directory) == FileAttributes.Directory;
            } catch {
                return false;
            }
        }

        static string CurrentFolder(IFileOpenDialog dialog) {
            try {
                IShellItem folder;
                dialog.GetFolder(out folder);
                string current;
                folder.GetDisplayName(SIGDN.FILESYSPATH, out current);
                return current ?? "";
            } catch {
                return "";
            }
        }

        static void CollectResults(IFileOpenDialog dialog, List<string> paths) {
            IShellItemArray items;
            dialog.GetResults(out items);
            ReadItems(items, paths);
        }

        static bool FocusIsTyping() {
            IntPtr focus = GetFocus();
            IntPtr cur = focus;
            for (int i = 0; i < 14 && cur != IntPtr.Zero; i++) {
                if (cur == _dialogHwnd) break;
                int id = GetDlgCtrlID(cur);
                if (id == 1148) return true;
                string cls = ClassNameOf(cur);
                if (cls == "Breadcrumb Parent" || cls == "Address Band Root" || cls == "TravelBand"
                    || cls == "AddressDisplay Control" || cls == "UniversalSearchBand"
                    || cls == "SearchBox" || cls == "ModernSearchBox" || cls == "SearchEditBox") {
                    return true;
                }
                if (cls == "Edit" || cls == "ComboBox" || cls == "ComboBoxEx32" || cls == "ToolbarWindow32") {
                    if (id == 1148 || HasAncestor(cur, "Address Band Root") || HasAncestor(cur, "Breadcrumb Parent") || HasAncestor(cur, "TravelBand")) {
                        return true;
                    }
                }
                cur = GetParent(cur);
            }
            return false;
        }

        static bool HasAncestor(IntPtr hwnd, string className) {
            IntPtr cur = GetParent(hwnd);
            for (int i = 0; i < 12 && cur != IntPtr.Zero; i++) {
                if (ClassNameOf(cur) == className) return true;
                if (cur == _dialogHwnd) return false;
                cur = GetParent(cur);
            }
            return false;
        }

        public static string ClassNameOf(IntPtr hwnd) {
            if (hwnd == IntPtr.Zero) return "";
            StringBuilder sb = new StringBuilder(256);
            if (GetClassName(hwnd, sb, sb.Capacity) <= 0) return "";
            return sb.ToString();
        }

        static bool Inside(IntPtr hwnd, IntPtr lParam) {
            int packed = lParam.ToInt32();
            short x = (short)(packed & 0xFFFF);
            short y = (short)((packed >> 16) & 0xFFFF);
            RECT rc;
            if (!GetClientRect(hwnd, out rc)) return true;
            return x >= 0 && y >= 0 && x < rc.Right && y < rc.Bottom;
        }

        static Form CreateOwner() {
            Form owner = new Form();
            owner.FormBorderStyle = FormBorderStyle.None;
            owner.ShowInTaskbar = false;
            owner.StartPosition = FormStartPosition.Manual;
            owner.TopMost = true;
            owner.MinimumSize = new Size(0, 0);
            owner.MaximumSize = new Size(2, 2);
            owner.ClientSize = new Size(1, 1);
            owner.Opacity = 0.01;
            owner.Text = "";
            Rectangle work = Screen.PrimaryScreen.WorkingArea;
            try {
                IntPtr fg = GetForegroundWindow();
                RECT wr;
                if (fg != IntPtr.Zero && GetWindowRect(fg, out wr)) {
                    Screen screen = Screen.FromRectangle(Rectangle.FromLTRB(wr.Left, wr.Top, wr.Right, wr.Bottom));
                    if (screen != null) work = screen.WorkingArea;
                }
            } catch { }
            int dialogWidth = 1200;
            int dialogHeight = 720;
            if (dialogWidth > work.Width) dialogWidth = work.Width;
            if (dialogHeight > work.Height) dialogHeight = work.Height;
            int x = work.Left + Math.Max(0, (work.Width - dialogWidth) / 2);
            int y = work.Top + Math.Max(0, (work.Height - dialogHeight) / 2);
            owner.Location = new Point(x, y);
            return owner;
        }

        static void CenterDialog(IntPtr hwnd) {
            RECT wr;
            if (!GetWindowRect(hwnd, out wr)) return;
            int width = wr.Right - wr.Left;
            int height = wr.Bottom - wr.Top;
            if (width < 200 || height < 200) return;
            Rectangle work = Screen.PrimaryScreen.WorkingArea;
            try {
                work = Screen.FromRectangle(Rectangle.FromLTRB(wr.Left, wr.Top, wr.Right, wr.Bottom)).WorkingArea;
            } catch { }
            int x = work.Left + Math.Max(0, (work.Width - width) / 2);
            int y = work.Top + Math.Max(0, (work.Height - height) / 2);
            SetWindowPos(hwnd, new IntPtr(-1), x, y, width, height, SWP_SHOWWINDOW);
        }

        static void ForceForeground(IntPtr hwnd) {
            if (hwnd == IntPtr.Zero) return;
            IntPtr fg = GetForegroundWindow();
            uint pid;
            uint fgThread = fg == IntPtr.Zero ? 0 : GetWindowThreadProcessId(fg, out pid);
            uint mine = GetCurrentThreadId();
            bool attached = false;
            if (fgThread != 0 && fgThread != mine) {
                attached = AttachThreadInput(mine, fgThread, true);
            }
            SetForegroundWindow(hwnd);
            BringWindowToTop(hwnd);
            if (attached) AttachThreadInput(mine, fgThread, false);
        }

        static void ActivateComctl() {
            if (_comctlReady) return;
            try {
                string dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HopBox");
                Directory.CreateDirectory(dir);
                string manifest = Path.Combine(dir, "comctl6.manifest");
                if (!File.Exists(manifest)) {
                    File.WriteAllText(manifest,
                        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>" +
                        "<assembly xmlns=\"urn:schemas-microsoft-com:asm.v1\" manifestVersion=\"1.0\">" +
                        "<dependency><dependentAssembly>" +
                        "<assemblyIdentity type=\"win32\" name=\"Microsoft.Windows.Common-Controls\" version=\"6.0.0.0\" processorArchitecture=\"*\" publicKeyToken=\"6595b64144ccf1df\" language=\"*\" />" +
                        "</dependentAssembly></dependency></assembly>");
                }
                ACTCTX act = new ACTCTX();
                act.cbSize = Marshal.SizeOf(typeof(ACTCTX));
                act.lpSource = manifest;
                _actCtx = CreateActCtx(ref act);
                if (_actCtx != new IntPtr(-1)) {
                    IntPtr cookie;
                    ActivateActCtx(_actCtx, out cookie);
                }
                INITCOMMONCONTROLSEX icc = new INITCOMMONCONTROLSEX();
                icc.dwSize = Marshal.SizeOf(typeof(INITCOMMONCONTROLSEX));
                icc.dwICC = 0x0000FFFF;
                InitCommonControlsEx(ref icc);
            } catch (Exception ex) {
                Debug("comctl " + ex.Message);
            }
            _comctlReady = true;
        }

        static bool TrySubclass(IntPtr hwnd, SubclassProc proc, UIntPtr id) {
            try {
                return SetWindowSubclass(hwnd, proc, id, UIntPtr.Zero);
            } catch (Exception ex) {
                Debug("SetWindowSubclass " + ex.Message);
                return false;
            }
        }

        static bool TryClassic(IntPtr hwnd, WndProc proc, out IntPtr old) {
            old = IntPtr.Zero;
            try {
                IntPtr thunk = Marshal.GetFunctionPointerForDelegate(proc);
                old = ReplaceProc(hwnd, thunk);
                return old != IntPtr.Zero;
            } catch (Exception ex) {
                Debug("classic " + ex.Message);
                return false;
            }
        }

        static IntPtr ReplaceProc(IntPtr hwnd, IntPtr proc) {
            if (IntPtr.Size == 8) return SetWindowLongPtr64(hwnd, GWLP_WNDPROC, proc);
            return SetWindowLong32(hwnd, GWLP_WNDPROC, proc);
        }

        static void ResetLog() {
            try {
                if (Environment.GetEnvironmentVariable("HOPBOX_PICK_DEBUG") != "1") return;
                string path = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HopBox", "dialog.log");
                if (File.Exists(path)) File.Delete(path);
            } catch { }
        }

        static void Debug(string line) {
            try {
                if (Environment.GetEnvironmentVariable("HOPBOX_PICK_DEBUG") != "1") return;
                string dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HopBox");
                Directory.CreateDirectory(dir);
                File.AppendAllText(Path.Combine(dir, "dialog.log"), DateTime.Now.ToString("HH:mm:ss.fff") + " " + line + "\r\n");
            } catch { }
        }

        [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
        static extern void SHCreateItemFromParsingName(string path, IntPtr bind, ref Guid iid, out IShellItem item);

        [DllImport("user32.dll")]
        static extern bool SetProcessDpiAwarenessContext(IntPtr value);
        [DllImport("user32.dll")]
        static extern IntPtr SetThreadDpiAwarenessContext(IntPtr value);
        [DllImport("shcore.dll")]
        static extern int SetProcessDpiAwareness(int value);
        [DllImport("kernel32.dll")]
        static extern uint GetCurrentThreadId();
        [DllImport("user32.dll")]
        static extern IntPtr GetDlgItem(IntPtr hDlg, int nIDDlgItem);
        [DllImport("user32.dll")]
        static extern bool IsWindow(IntPtr hWnd);
        [DllImport("user32.dll")]
        static extern bool IsWindowVisible(IntPtr hWnd);
        [DllImport("user32.dll", EntryPoint = "EnumWindows")]
        static extern bool EnumWindowsEntry(EnumProc proc, IntPtr lparam);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
        delegate bool EnumProc(IntPtr hwnd, IntPtr lparam);
        [DllImport("user32.dll")]
        static extern IntPtr GetFocus();
        [DllImport("user32.dll")]
        static extern IntPtr GetParent(IntPtr hWnd);
        [DllImport("user32.dll")]
        static extern int GetDlgCtrlID(IntPtr hWnd);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        static extern int GetClassName(IntPtr hWnd, StringBuilder lpClassName, int nMaxCount);
        [DllImport("user32.dll")]
        static extern bool GetClientRect(IntPtr hWnd, out RECT lpRect);
        [DllImport("user32.dll")]
        static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
        [DllImport("user32.dll")]
        static extern bool PostMessage(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
        [DllImport("user32.dll")]
        static extern bool ReleaseCapture();
        [DllImport("user32.dll")]
        static extern void PostQuitMessage(int nExitCode);
        [DllImport("user32.dll")]
        static extern bool SetForegroundWindow(IntPtr hWnd);
        [DllImport("user32.dll")]
        static extern IntPtr GetForegroundWindow();
        [DllImport("user32.dll")]
        static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
        [DllImport("user32.dll")]
        static extern bool AttachThreadInput(uint idAttach, uint idAttachTo, bool fAttach);
        [DllImport("user32.dll")]
        static extern bool BringWindowToTop(IntPtr hWnd);
        [DllImport("user32.dll")]
        static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int x, int y, int cx, int cy, uint flags);
        [DllImport("user32.dll", EntryPoint = "SetWindowLongPtrW", SetLastError = true)]
        static extern IntPtr SetWindowLongPtr64(IntPtr hWnd, int nIndex, IntPtr dwNewLong);
        [DllImport("user32.dll", EntryPoint = "SetWindowLongW", SetLastError = true)]
        static extern IntPtr SetWindowLong32(IntPtr hWnd, int nIndex, IntPtr dwNewLong);
        [DllImport("user32.dll")]
        static extern IntPtr CallWindowProc(IntPtr lpPrevWndFunc, IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
        [DllImport("user32.dll", SetLastError = true)]
        static extern IntPtr SetWindowsHookEx(int idHook, HookProc lpfn, IntPtr hMod, uint dwThreadId);
        [DllImport("user32.dll")]
        static extern bool UnhookWindowsHookEx(IntPtr hhk);
        [DllImport("user32.dll")]
        static extern IntPtr CallNextHookEx(IntPtr hhk, int nCode, IntPtr wParam, IntPtr lParam);
        [DllImport("comctl32.dll", SetLastError = true)]
        static extern bool SetWindowSubclass(IntPtr hWnd, SubclassProc pfnSubclass, UIntPtr uIdSubclass, UIntPtr dwRefData);
        [DllImport("comctl32.dll")]
        static extern bool RemoveWindowSubclass(IntPtr hWnd, SubclassProc pfnSubclass, UIntPtr uIdSubclass);
        [DllImport("comctl32.dll")]
        static extern IntPtr DefSubclassProc(IntPtr hWnd, uint uMsg, IntPtr wParam, IntPtr lParam);
        [DllImport("comctl32.dll")]
        static extern bool InitCommonControlsEx(ref INITCOMMONCONTROLSEX icc);
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        static extern IntPtr CreateActCtx(ref ACTCTX actctx);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool ActivateActCtx(IntPtr hActCtx, out IntPtr lpCookie);

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        delegate IntPtr SubclassProc(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam, UIntPtr id, UIntPtr data);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        delegate IntPtr WndProc(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        delegate IntPtr HookProc(int nCode, IntPtr wParam, IntPtr lParam);

        [StructLayout(LayoutKind.Sequential)]
        public struct POINT { public int X; public int Y; }
        [StructLayout(LayoutKind.Sequential)]
        public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }
        [StructLayout(LayoutKind.Sequential)]
        struct INITCOMMONCONTROLSEX { public int dwSize; public int dwICC; }
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        struct ACTCTX {
            public int cbSize;
            public uint dwFlags;
            public string lpSource;
            public ushort wProcessorArchitecture;
            public ushort wLangId;
            public string lpAssemblyDirectory;
            public string lpResourceName;
            public string lpApplicationName;
            public IntPtr hModule;
        }

        [Flags]
        enum FOS : uint {
            PICKFOLDERS = 0x20,
            FORCEFILESYSTEM = 0x40,
            NOVALIDATE = 0x100,
            ALLOWMULTISELECT = 0x200,
            PATHMUSTEXIST = 0x800,
            DONTADDTORECENT = 0x2000000,
        }
        enum SIGDN : uint { FILESYSPATH = 0x80058000 }
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        struct COMDLG_FILTERSPEC { public string Name; public string Spec; }

        [ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")]
        class FileOpenDialogRCW { }

        [ComImport, Guid("00000114-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IOleWindow {
            void GetWindow(out IntPtr phwnd);
            void ContextSensitiveHelp([MarshalAs(UnmanagedType.Bool)] bool fEnterMode);
        }

        [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IShellItem {
            void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
            void GetParent(out IShellItem ppsi);
            void GetDisplayName(SIGDN sigdn, [MarshalAs(UnmanagedType.LPWStr)] out string name);
            void GetAttributes(uint mask, out uint attrs);
            void Compare(IShellItem other, uint hint, out int order);
        }

        [ComImport, Guid("b63ea76d-1f85-456f-a19c-48159efa858b"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IShellItemArray {
            void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
            void GetPropertyStore(int flags, ref Guid riid, out IntPtr ppv);
            void GetPropertyDescriptionList(ref Guid key, ref Guid riid, out IntPtr ppv);
            void GetAttributes(int flags, uint mask, out uint attrs);
            void GetCount(out uint count);
            void GetItemAt(uint index, out IShellItem item);
            void EnumItems(out IntPtr enumerator);
        }

        [ComImport, Guid("973510db-7d7f-452b-8975-74a85828d354"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IFileDialogEvents {
            [PreserveSig] int OnFileOk(IFileOpenDialog pfd);
            [PreserveSig] int OnFolderChanging(IFileOpenDialog pfd, IShellItem psiFolder);
            [PreserveSig] int OnFolderChange(IFileOpenDialog pfd);
            [PreserveSig] int OnSelectionChange(IFileOpenDialog pfd);
            [PreserveSig] int OnShareViolation(IFileOpenDialog pfd, IShellItem psi, out int response);
            [PreserveSig] int OnTypeChange(IFileOpenDialog pfd);
            [PreserveSig] int OnOverwrite(IFileOpenDialog pfd, IShellItem psi, out int response);
        }

        [ComImport, Guid("d57c7288-d4ad-4768-be02-9d969532d960"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IFileOpenDialog {
            [PreserveSig] int Show(IntPtr parent);
            void SetFileTypes(uint count, [MarshalAs(UnmanagedType.LPArray)] COMDLG_FILTERSPEC[] spec);
            void SetFileTypeIndex(uint index);
            void GetFileTypeIndex(out uint index);
            void Advise(IFileDialogEvents events, out uint cookie);
            void Unadvise(uint cookie);
            void SetOptions(FOS options);
            void GetOptions(out FOS options);
            void SetDefaultFolder(IShellItem item);
            void SetFolder(IShellItem item);
            void GetFolder(out IShellItem item);
            void GetCurrentSelection(out IShellItem item);
            void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
            void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
            void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
            void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
            void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
            void GetResult(out IShellItem item);
            void AddPlace(IShellItem item, int placement);
            void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
            void Close(int hr);
            void SetClientGuid(ref Guid guid);
            void ClearClientData();
            void SetFilter(IntPtr filter);
            void GetResults(out IShellItemArray items);
            void GetSelectedItems(out IShellItemArray items);
        }

        [ComImport, Guid("6d5140c1-7436-11ce-8034-00aa006009fa"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IShellService {
            [PreserveSig] int QueryService(ref Guid guidService, ref Guid riid, out IntPtr ppv);
        }

        [ComImport, Guid("000214E3-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IShellView { }

        [ComImport, Guid("000214E2-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IShellBrowser {
            void GetWindow(out IntPtr phwnd);
            void ContextSensitiveHelp([MarshalAs(UnmanagedType.Bool)] bool fEnterMode);
            void InsertMenusSB(IntPtr hmenuShared, IntPtr lpMenuWidths);
            void SetMenuSB(IntPtr hmenuShared, IntPtr holemenuRes, IntPtr hwndActiveObject);
            void RemoveMenusSB(IntPtr hmenuShared);
            void SetStatusTextSB(IntPtr pszStatusText);
            void EnableModelessSB([MarshalAs(UnmanagedType.Bool)] bool fEnable);
            void TranslateAcceleratorSB(IntPtr pmsg, ushort wID);
            void BrowseObject(IntPtr pidl, uint wFlags);
            void GetViewStateStream(uint grfMode, out IntPtr ppStrm);
            void GetControlWindow(uint id, out IntPtr phwnd);
            void SendControlMsg(uint id, uint uMsg, IntPtr wParam, IntPtr lParam, out IntPtr pret);
            void QueryActiveShellView(out IShellView ppshv);
        }

        [ComImport, Guid("cde725b0-ccc9-4519-917e-325d72fab4ce"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IFolderView {
            void GetCurrentViewMode(out uint pViewMode);
            void SetCurrentViewMode(uint viewMode);
            void GetFolder(ref Guid riid, out IntPtr ppv);
            void Item(int iItemIndex, out IntPtr ppidl);
            void ItemCount(uint uFlags, out int pcItems);
            [PreserveSig] int Items(uint uFlags, ref Guid riid, out IntPtr ppv);
        }

        [ComVisible(true)]
        [ClassInterface(ClassInterfaceType.None)]
        class ImportEvents : IFileDialogEvents {
            public int OnFileOk(IFileOpenDialog pfd) {
                Install(pfd);
                return 0;
            }
            public int OnFolderChanging(IFileOpenDialog pfd, IShellItem psiFolder) {
                return 0;
            }
            public int OnFolderChange(IFileOpenDialog pfd) {
                Install(pfd);
                return 0;
            }
            public int OnSelectionChange(IFileOpenDialog pfd) {
                Install(pfd);
                return 0;
            }
            public int OnShareViolation(IFileOpenDialog pfd, IShellItem psi, out int response) {
                response = 0;
                return 0;
            }
            public int OnTypeChange(IFileOpenDialog pfd) {
                Install(pfd);
                return 0;
            }
            public int OnOverwrite(IFileOpenDialog pfd, IShellItem psi, out int response) {
                response = 0;
                return 0;
            }
        }
    }
}
