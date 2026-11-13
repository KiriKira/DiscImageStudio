using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using DiscImageStudio.Controls;
using DiscImageStudio.Core;
using DiscImageStudio.ImageIO;
using DiscImageStudio.Localization;
using DiscImageStudio.ViewModels;
using DvdImageSolver.PhysicalImaging;
using DvdImageSolver.Solver;

internal static class PhysicalPreviewFlowTests
{
    internal static void Run()
    {
        var directory = Directory.CreateTempSubdirectory("physical-preview-flow-");
        try { WpfTestHost.Run(() => Wait(CheckAsync(directory.FullName))); }
        finally { directory.Delete(recursive: true); }
        Console.WriteLine("physical-preview-flow: passed (enabled refresh button, manual conversion, profile/error recovery, prepared calibration pixels, measured geometry, legacy restore, project close)");
    }

    private static async Task CheckAsync(string directory)
    {
        string language = L.CurrentLanguage;
        L.SetLanguage(L.Chinese);
        var shell = new ShellViewModel(Dispatcher.CurrentDispatcher);
        try
        {
            string source = Path.Combine(directory, "midgray.png");
            RasterCodec.SavePng(source, 65, 65, Enumerable.Repeat((byte)128, 65 * 65).ToArray(), true);
            byte[] original = SHA256.HashData(File.ReadAllBytes(source));
            var document = new ProjectDocument(ProjectJsonStore.CurrentSchemaVersion, "Physical preview regression", ProjectDocument.DvdFamily,
                new(24.5, 58, 4096, UsableSectors: 4096), [new("layer", "Mid-gray", source, true, 0, 0, 0, 1)],
                new(Path.Combine(directory, "disc.iso"), false, "", "PHYSICAL_TEST", false),
                null, null, DateTimeOffset.UnixEpoch, DateTimeOffset.UnixEpoch, ToneProfile.LegacyBinary);
            typeof(ShellViewModel).GetProperty(nameof(ShellViewModel.CurrentProject))!.SetValue(shell, document);
            shell.IsProjectOpen = true;
            shell.Design.SetDiscFamily(OpticalDiscFamily.Dvd);
            ProjectSession.ApplyToState(document, shell.State);
            shell.SelectWorkspace(Workspace.Design);
            shell.Design.LoadLayers(document.Layers);
            ToneViewModel tone = shell.Design.Tone;
            await Until(() => tone.IsTonePreviewReady, () => tone.TonePreviewStatus);
            string composite = shell.State.DvdImagePath;
            byte[] compositeHash = SHA256.HashData(File.ReadAllBytes(composite));
            var settings = new DvdPhysicalConversionSettings
            {
                CellUm = 500, MapSize = 17, Passes = 0,
                Views = [new("Test", 0, 0, 800, [new(Samples: 1)])]
            };
            shell.State.DvdPhysicalProfileJson = JsonSerializer.Serialize(settings, DvdPhysicalConversion.JsonOptions);
            shell.State.DvdPhysicalCellUm = 500;
            shell.State.DvdPhysicalEnabled = true;
            Require(!tone.HasTonePreview && tone.RefreshTonePreviewCommand.CanExecute(null),
                "enabling physical mode discards old pixels but keeps manual refresh available");
            var panel = new TonePanel { DataContext = tone };
            panel.Measure(new Size(344, 920)); panel.Arrange(new Rect(0, 0, 344, 920)); panel.UpdateLayout();
            Dispatcher.CurrentDispatcher.Invoke(() => { }, DispatcherPriority.DataBind);
            Require(panel.FindName("RefreshTonePreviewButton") is Button { IsEnabled: true },
                "the actual refresh button remains enabled under physical controls");
            shell.State.DvdPhysicalCellUm = 400;
            Require(tone.RefreshTonePreviewCommand.CanExecute(null), "grid edits retain the input path");
            await RefreshTone(tone);
            Require(tone.HasTonePreview && tone.TonePreviewStatus.Contains("物理"), "manual calculation publishes physical pixels and status");
            panel.Measure(new Size(344, 920)); panel.Arrange(new Rect(0, 0, 344, 920)); panel.UpdateLayout();
            UiLayoutSnapshot.Save(panel, "physical-tone-ready");

            // Invalid profiles must not leave the button disabled or an obsolete image displayed.
            string profile = shell.State.DvdPhysicalProfileJson;
            shell.State.DvdPhysicalProfileJson = "{";
            tone.RefreshTonePreviewCommand.Execute(null);
            await Until(() => tone.TonePreviewStatus.Contains("失败"), () => tone.TonePreviewStatus);
            Require(!tone.HasTonePreview && tone.RefreshTonePreviewCommand.CanExecute(null), "invalid profile can be corrected and refreshed");
            shell.State.DvdPhysicalProfileJson = profile;
            await RefreshTone(tone);

            using var cancellation = new CancellationTokenSource();
            string prepared = await shell.Design.EnsureExportCompositeAsync(cancellation.Token);
            string density = Path.Combine(Path.GetDirectoryName(prepared)!, "density.png");
            Require(Pixels(tone.TonePreviewImage!).SequenceEqual(Pixels(Load(density))), "design displays the prepared density preview");

            LivePreviewViewModel preview = shell.LivePreview;
            preview.SelectedFamily = OpticalDiscFamily.Dvd;
            preview.IsCalibrationPreview = true;
            shell.State.DvdActualInnerRadius = "24.5";
            shell.State.DvdActualOuterRadius = "58";
            shell.State.DvdActualPitchLinear = shell.State.DvdActualPitchQuadratic = shell.State.DvdActualPitchCubic = "0";
            shell.State.DvdPreviewSize = "256"; shell.State.DvdSamplesPerSector = "16";
            shell.SelectWorkspace(Workspace.Preview);
            await RefreshLive(preview);
            Require(preview.HasResultImage && preview.StatusText.Contains("物理抖动") && !preview.StatusText.Contains("抖动 0%"),
                "calibration identifies prepared physical dithering rather than the disabled legacy slider");
            byte[] first = Pixels(preview.ResultImage!);
            CompareProjection(preview, prepared, directory);
            shell.State.DvdActualOuterRadius = "58.5";
            await RefreshLive(preview);
            CompareProjection(preview, prepared, directory);
            Require(!first.SequenceEqual(Pixels(preview.ResultImage!)), "measured geometry still warps the physical pattern");
            Require(preview.ResultPreviewPath == composite && shell.State.DvdImagePath == composite,
                "the temporary physical cache never replaces the source shown to the user");

            shell.State.DvdPhysicalEnabled = false;
            await RefreshLive(preview);
            Require(preview.StatusText.Contains("抖动 0%"), "disabling physical mode restores the legacy tone path");
            Require(SHA256.HashData(File.ReadAllBytes(composite)).SequenceEqual(compositeHash)
                && SHA256.HashData(File.ReadAllBytes(source)).SequenceEqual(original), "all previews preserve original and composed sources");

            // Enable before a new composite arrives, including an imported project's path.
            shell.SelectWorkspace(Workspace.Design);
            shell.State.DvdPhysicalEnabled = true;
            bool receivedComposite = false;
            void CompositeReady(string? path) => receivedComposite = path is not null;
            shell.Design.CompositeRendered += CompositeReady;
            try
            {
                shell.Design.LoadLayers(document.Layers);
                await Until(() => receivedComposite, () => tone.TonePreviewStatus);
            }
            finally { shell.Design.CompositeRendered -= CompositeReady; }
            Require(tone.RefreshTonePreviewCommand.CanExecute(null), "physical mode accepts a newly imported composite");
            await RefreshTone(tone);
            tone.RefreshTonePreviewCommand.Execute(null);
            shell.IsProjectOpen = false;
            shell.Design.ClearLayers();
            await Task.Delay(250);
            Require(!tone.RefreshTonePreviewCommand.CanExecute(null) && !tone.HasTonePreview && tone.TonePreviewImage is null,
                "closing the project clears the input and prevents late publication");
        }
        finally
        {
            shell.IsProjectOpen = false; shell.Design.ClearLayers(); shell.LivePreview.OnWindowClosed();
            L.SetLanguage(language);
        }
    }

    private static async Task RefreshTone(ToneViewModel tone)
    {
        Require(tone.RefreshTonePreviewCommand.CanExecute(null), "manual refresh executable");
        tone.RefreshTonePreviewCommand.Execute(null);
        await Until(() => tone.IsTonePreviewReady, () => tone.TonePreviewStatus);
    }

    private static async Task RefreshLive(LivePreviewViewModel preview)
    {
        BitmapSource? previous = preview.ResultImage;
        preview.RefreshCommand.Execute(null);
        await Until(() => preview.IsLivePreviewReady && preview.ResultImage is not null
            && !ReferenceEquals(previous, preview.ResultImage), () => preview.StatusText);
    }

    private static void CompareProjection(LivePreviewViewModel preview, string prepared, string directory)
    {
        object?[] args = [null, null, null, null];
        bool ok = (bool)typeof(LivePreviewViewModel).GetMethod("TryBuildDvdCalibrationOptions", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(preview, args)!;
        Require(ok && args[1] is CalibrationPreviewOptions { PhysicalBinary: true }, "calibration explicitly preserves physical binary sampling");
        string expected = Path.Combine(directory, "expected.png");
        CalibrationPreviewRenderer.Render(prepared, expected, (CalibrationPreviewOptions)args[1]!);
        Require(Pixels(preview.ResultImage!).SequenceEqual(Pixels(Load(expected))), "calibration projects the exact same prepared pattern as export");
    }

    private static BitmapSource Load(string path)
    {
        var bitmap = new BitmapImage();
        using var stream = File.OpenRead(path);
        bitmap.BeginInit(); bitmap.CacheOption = BitmapCacheOption.OnLoad; bitmap.StreamSource = stream; bitmap.EndInit(); bitmap.Freeze();
        return bitmap;
    }

    private static byte[] Pixels(BitmapSource image)
    {
        var bitmap = new FormatConvertedBitmap(image, PixelFormats.Bgra32, null, 0);
        var bytes = new byte[bitmap.PixelWidth * bitmap.PixelHeight * 4]; bitmap.CopyPixels(bytes, bitmap.PixelWidth * 4, 0); return bytes;
    }

    private static async Task Until(Func<bool> condition, Func<string> status)
    {
        var timer = Stopwatch.StartNew();
        while (!condition())
        {
            if (timer.Elapsed > TimeSpan.FromSeconds(90)) throw new InvalidOperationException("Physical preview timeout: " + status());
            await Task.Delay(25);
        }
    }

    private static void Wait(Task task)
    {
        var frame = new DispatcherFrame(); var dispatcher = Dispatcher.CurrentDispatcher;
        _ = task.ContinueWith(_ => dispatcher.BeginInvoke(new Action(() => frame.Continue = false)), TaskScheduler.Default);
        if (!task.IsCompleted) Dispatcher.PushFrame(frame);
        task.GetAwaiter().GetResult();
    }
    private static void Require(bool condition, string message)
    { if (!condition) throw new InvalidOperationException("Physical preview flow: " + message); }
}
