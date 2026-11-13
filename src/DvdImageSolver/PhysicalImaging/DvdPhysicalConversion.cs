using System.Security.Cryptography;
using System.Text.Json;
using DiscImageStudio.ImageIO;

namespace DvdImageSolver.PhysicalImaging;

public sealed record PhysicalWavelength(double Nm, double Weight);
public sealed record PhysicalTexture(double Diffuse, double Specular, double Diffraction,
    double SpecularWidth = .045, double RadialWidth = .06, double TangentWidth = .035);
public sealed record PhysicalLight(double X = 135, double Y = 0, double Z = 400,
    double Width = 30, double Height = 30, int Samples = 9, double Radiance = 1);
public sealed record PhysicalView(string Name, double CameraX, double CameraY, double CameraZ,
    PhysicalLight[] Lights, double Weight = 1, PhysicalWavelength[]? Wavelengths = null);
public sealed record DvdPhysicalConversionSettings
{
    public int Version { get; init; } = 1;
    public string Provenance { get; init; } = "hypothetical";
    public string? Evidence { get; init; }
    public string Recipe { get; init; } = "off";
    public double CellUm { get; init; } = 20;
    public int MapSize { get; init; } = 97;
    public int Passes { get; init; } = 4;
    public double ToneExponent { get; init; } = 1;
    // Photo density is perceptual by default; linear radiance is an explicit calibration choice.
    public string ToneEncoding { get; init; } = "perceptual";
    public double ReflectanceCompensation { get; init; } = 0;
    public double MaximumToneShift { get; init; } = .08;
    public uint Seed { get; init; } = 43523;
    public PhysicalWavelength[] Wavelengths { get; init; } = [new(450, .2), new(550, .6), new(650, .2)];
    public PhysicalTexture[] Textures { get; init; } = [new(.025, .30, .12), new(.08, .35, .08)];
    public PhysicalView[] Views { get; init; } = [
        new("Right lamp", 0, 0, 800, [new()]),
        new("Top lamp", 0, 0, 800, [new(0, 135)]),
        new("Camera offset 2 degrees", 27.94, 0, 800, [new()])];

    public void Validate()
    {
        static bool Positive(double x) => double.IsFinite(x) && x > 0;
        if (Version != 1 || Provenance is not ("hypothetical" or "measured")
            || Provenance == "measured" && string.IsNullOrWhiteSpace(Evidence)
            || ToneEncoding is not ("perceptual" or "linear")
            || !double.IsFinite(ReflectanceCompensation) || ReflectanceCompensation is < 0 or > 1
            || !double.IsFinite(MaximumToneShift) || MaximumToneShift is < 0 or > 1
            || !Positive(CellUm) || !Positive(ToneExponent) || MapSize is < 17 or > 513 || Passes is < 0 or > 4
            || Recipe is not ("off" or "periodic-9b-16-v1") || Textures is not { Length: 2 }
            || Wavelengths is not { Length: >= 1 and <= 32 } || Views is not { Length: >= 1 and <= 8 } || Views.Any(v => v is null))
            throw new ArgumentException("Invalid physical conversion settings or provenance.");
        foreach (var w in Wavelengths)
            if (w is null || !Positive(w.Nm) || !Positive(w.Weight)) throw new ArgumentException("Invalid sensor-weighted spectrum.");
        foreach (var t in Textures)
            if (t is null || !double.IsFinite(t.Diffuse + t.Specular + t.Diffraction)
                || Math.Min(t.Diffuse, Math.Min(t.Specular, t.Diffraction)) < 0
                || t.Diffuse + t.Specular + t.Diffraction > 1
                || !Positive(t.SpecularWidth) || !Positive(t.RadialWidth) || !Positive(t.TangentWidth))
                throw new ArgumentException("Physical textures must be passive with positive lobe widths.");
        if (!double.IsFinite(Wavelengths.Sum(w => w.Weight)) || !double.IsFinite(Views.Sum(v => v.Weight)))
            throw new ArgumentException("Physical condition weights overflow.");
        foreach (var v in Views)
        {
            if (v is null || !Positive(v.CameraZ) || !Positive(v.Weight)
                || !double.IsFinite(v.CameraX + v.CameraY) || v.Lights is not { Length: >= 1 and <= 16 })
                throw new ArgumentException("Invalid physical viewing condition.");
            if (v.Wavelengths is { } spectrum && (spectrum.Length is < 1 or > 32 || spectrum.Any(w => w is null || !Positive(w.Nm) || !Positive(w.Weight))
                || !double.IsFinite(spectrum.Sum(w => w.Weight)))) throw new ArgumentException("Invalid viewing-condition spectrum.");
            foreach (var l in v.Lights)
                if (l is null || !Positive(l.Z) || !Positive(l.Width) || !Positive(l.Height)
                    || !Positive(l.Radiance) || !double.IsFinite(l.X + l.Y) || l.Samples is < 1 or > 32)
                    throw new ArgumentException("Invalid physical area light.");
        }
    }
}

public sealed record DvdPhysicalConversionRequest(string SourceImage, double CanvasRadiusMm,
    double InnerRadiusMm, double OuterRadiusMm, DvdPhysicalConversionSettings Settings,
    double TrackPitchNm = 740, double? DrawingInnerRadiusMm = null, double? DrawingOuterRadiusMm = null,
    double PitchLinear = 0, double PitchQuadratic = 0, double PitchCubic = 0, double? GeometryOuterRadiusMm = null);
public sealed record DvdPhysicalConversionResult(string BinaryPath, string ReportPath, string ContentHash,
    int Size, double CellUm, double ClippedFraction, double UnobservableFraction, string? MaskedPath = null);

public static class DvdPhysicalConversion
{
    public const string Algorithm = "dvd-physical-density-v3.1";
    public static JsonSerializerOptions JsonOptions { get; } = new()
    { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, PropertyNameCaseInsensitive = true, WriteIndented = true,
      UnmappedMemberHandling = System.Text.Json.Serialization.JsonUnmappedMemberHandling.Disallow };

    public static double Brdf(PhysicalTexture t, double sx, double sy, double vx, double vy,
        double ex, double ey, double wavelength, double pitch)
    {
        double qr = (sx + vx) * ex + (sy + vy) * ey;
        double qt = -(sx + vx) * ey + (sy + vy) * ex;
        static double K(double r, double a, double wr, double wa)
            => Math.Exp(-.5 * (r * r / (wr * wr) + a * a / (wa * wa))) / (2 * Math.PI * wr * wa);
        return t.Diffuse / Math.PI + t.Specular * K(qr, qt, t.SpecularWidth, t.SpecularWidth)
            + .5 * t.Diffraction * (K(qr - wavelength / pitch, qt, t.RadialWidth, t.TangentWidth)
                + K(qr + wavelength / pitch, qt, t.RadialWidth, t.TangentWidth));
    }

    public static DvdPhysicalConversionResult Prepare(DvdPhysicalConversionRequest request, string cacheDirectory,
        Action<string, double>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Settings);
        cacheDirectory = Path.GetFullPath(cacheDirectory);
        request.Settings.Validate();
        double radius = request.CanvasRadiusMm, inner = request.DrawingInnerRadiusMm ?? request.InnerRadiusMm,
            outer = request.DrawingOuterRadiusMm ?? request.OuterRadiusMm;
        if (!double.IsFinite(radius) || radius <= 0 || !double.IsFinite(inner) || !double.IsFinite(outer)
            || inner < 0 || inner >= outer || outer > radius || !double.IsFinite(request.TrackPitchNm) || request.TrackPitchNm <= 0)
            throw new ArgumentException("Invalid physical disc geometry.");
        int size = checked((int)Math.Round(radius * 2000 / request.Settings.CellUm) + 1);
        if (size is < 33 or > 8192) throw new ArgumentException("Physical conversion supports 33..8192 cells per axis.");
        cancellationToken.ThrowIfCancellationRequested();
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using var sourceLease = File.OpenRead(request.SourceImage);
        {
            var input = sourceLease;
            byte[] buffer = new byte[65536]; int read;
            while ((read = input.Read(buffer)) > 0) { cancellationToken.ThrowIfCancellationRequested(); hash.AppendData(buffer.AsSpan(0, read)); }
        }
        string sourceHash = Convert.ToHexString(hash.GetCurrentHash());
        hash.AppendData(System.Text.Encoding.UTF8.GetBytes(Algorithm + JsonSerializer.Serialize(request with { SourceImage = "" }, JsonOptions)));
        string key = Convert.ToHexString(hash.GetHashAndReset());
        string directory = Path.Combine(cacheDirectory, key);
        string binary = Path.Combine(directory, "binary.png"), report = Path.Combine(directory, "report.json");
        if (File.Exists(binary) && File.Exists(report))
        {
            var cached = JsonSerializer.Deserialize<DvdPhysicalConversionResult>(File.ReadAllText(Path.Combine(directory, "result.json")), JsonOptions);
            if (cached is not null) return cached with { BinaryPath = binary, ReportPath = report, MaskedPath = Path.Combine(directory, "masked.png") };
        }
        var raster = RasterCodec.LoadBgra32(request.SourceImage, maxPixels: 8192L * 8192);
        sourceLease.Dispose();
        if (raster.Width != raster.Height) throw new ArgumentException("Physical conversion requires a square disc-layout image.");
        var linearSource = new float[raster.Width * raster.Height];
        double[] sourceLut = Enumerable.Range(0, 256).Select(value => value / 255.0 <= .04045 ? value / 255.0 / 12.92 : Math.Pow((value / 255.0 + .055) / 1.055, 2.4)).ToArray();
        for (int i = 0; i < linearSource.Length; i++)
        {
            if ((i & 65535) == 0) cancellationToken.ThrowIfCancellationRequested();
            double alpha = raster.Pixels[i * 4 + 3] / 255.0;
            linearSource[i] = (float)(alpha * (.0722 * sourceLut[raster.Pixels[i * 4]] + .7152 * sourceLut[raster.Pixels[i * 4 + 1]] + .2126 * sourceLut[raster.Pixels[i * 4 + 2]]) + 1 - alpha);
        }
        var settings = request.Settings; int views = settings.Views.Length, m = settings.MapSize;
        float[][] maps = Enumerable.Range(0, views * 2).Select(_ => new float[m * m]).ToArray();
        for (int y = 0; y < m; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            progress?.Invoke("Optical response", .15 * y / m);
            for (int x = 0; x < m; x++)
            {
                double px = (2.0 * x / (m - 1) - 1) * radius, py = (1 - 2.0 * y / (m - 1)) * radius;
                double rr = Math.Sqrt(px * px + py * py), ex = rr == 0 ? 1 : px / rr, ey = rr == 0 ? 0 : py / rr;
                double u = Math.Clamp((rr - request.InnerRadiusMm) / Math.Max(1e-9, (request.GeometryOuterRadiusMm ?? request.OuterRadiusMm) - request.InnerRadiusMm), 0, 1);
                double localPitch = request.TrackPitchNm * (1 + request.PitchLinear * u + request.PitchQuadratic * u * u + request.PitchCubic * u * u * u);
                if (!double.IsFinite(localPitch) || localPitch <= 0) throw new ArgumentException("Invalid local track pitch.");
                for (int v = 0; v < views; v++)
                {
                    var view = settings.Views[v]; double dx = view.CameraX - px, dy = view.CameraY - py;
                    double cd = Math.Sqrt(dx * dx + dy * dy + view.CameraZ * view.CameraZ);
                    foreach (var light in view.Lights)
                        for (int a = 0; a < light.Samples; a++) for (int b = 0; b < light.Samples; b++)
                        {
                            double lx = light.X + ((a + .5) / light.Samples - .5) * light.Width - px;
                            double ly = light.Y + ((b + .5) / light.Samples - .5) * light.Height - py;
                            double d2 = lx * lx + ly * ly + light.Z * light.Z, d = Math.Sqrt(d2);
                            double energy = light.Radiance * light.Z * light.Z * light.Width * light.Height / (light.Samples * light.Samples * d2 * d2);
                            var spectrum = view.Wavelengths ?? settings.Wavelengths;
                            double spectrumWeight = spectrum.Sum(w => w.Weight);
                            for (int t = 0; t < 2; t++)
                                foreach (var wavelength in spectrum)
                                    maps[v * 2 + t][y * m + x] += (float)(energy * wavelength.Weight / spectrumWeight
                                        * Brdf(settings.Textures[t], lx / d, ly / d, dx / cd, dy / cd, ex, ey, wavelength.Nm, localPitch));
                        }
                }
            }
        }
        if (maps.Any(map => map.Any(value => !float.IsFinite(value))))
            throw new ArgumentException("Optical response overflow; check radiance and lobe widths.");
        double[] lo = new double[views], hi = new double[views], displayMax = new double[views];
        for (int v = 0; v < views; v++)
        {
            var medians = new double[2];
            for (int t = 0; t < 2; t++)
            {
                var samples = maps[v * 2 + t].Where((_, i) =>
                { double xx = (2.0 * (i % m) / (m - 1) - 1) * radius, yy = (2.0 * (i / m) / (m - 1) - 1) * radius;
                  return xx * xx + yy * yy >= inner * inner && xx * xx + yy * yy <= outer * outer; }).Order().ToArray();
                medians[t] = samples.Length == 0 ? 0 : samples[samples.Length / 2];
                if (samples.Length > 0) displayMax[v] = Math.Max(displayMax[v], samples[^1]);
            }
            lo[v] = medians.Min(); hi[v] = medians.Max();
        }
        var pixels = new byte[size * size];
        var diagnosticFlags = new byte[size * size];
        var tone = new float[size * size];
        var mask = new bool[size * size];
        double[] error = new double[size * 2];
        double[] delta = new double[views];
        (int X, int Y, int W)[] neighbors = new (int, int, int)[4];
        long clipped = 0, modelOutOfRange = 0, unobservable = 0, active = 0;
        double sourceDensitySum = 0, targetDensitySum = 0;
        double continuousError = 0;
        double weightSum = settings.Views.Sum(v => v.Weight);
        double cell = radius * 2 / (size - 1);
        for (int y = 0; y < size; y++)
        {
            cancellationToken.ThrowIfCancellationRequested(); progress?.Invoke("Photo density error diffusion", .15 + .35 * y / size);
            int dir = y % 2 == 0 ? 1 : -1;
            for (int ix = 0; ix < size; ix++)
            {
                int x = dir == 1 ? ix : size - 1 - ix, i = y * size + x;
                double px = x * cell - radius, py = radius - y * cell, rr2 = px * px + py * py;
                if (rr2 < inner * inner || rr2 > outer * outer) continue;
                int sourceOffset = ((int)Math.Round(y * (raster.Height - 1.0) / (size - 1)) * raster.Width
                    + (int)Math.Round(x * (raster.Width - 1.0) / (size - 1))) * 4;
                if (raster.Pixels[sourceOffset + 3] == 0) continue;
                mask[i] = true; active++;
                double luminance = ResampleSource(linearSource, raster.Width, x, y, size);
                double gray = Math.Pow(settings.ToneEncoding == "linear" ? luminance : EncodeSrgb(luminance), settings.ToneExponent);
                double numerator = 0, denominator = 0;
                for (int v = 0; v < views; v++)
                {
                    double r0 = Sample(maps[v * 2], m, x, y, size), r1 = Sample(maps[v * 2 + 1], m, x, y, size);
                    delta[v] = r1 - r0; double w = settings.Views[v].Weight / weightSum;
                    numerator += w * delta[v] * (lo[v] + (hi[v] - lo[v]) * gray - r0);
                    denominator += w * delta[v] * delta[v];
                }
                double raw = denominator < 1e-30 ? gray : numerator / denominator;
                if (denominator < 1e-30) unobservable++;
                if (raw < 0 || raw > 1) modelOutOfRange++;
                double requested = gray + settings.ReflectanceCompensation * Math.Clamp(raw - gray, -settings.MaximumToneShift, settings.MaximumToneShift);
                if (requested < 0 || requested > 1) clipped++;
                double p = Math.Clamp(requested, 0, 1);
                tone[i] = (float)p;
                sourceDensitySum += gray; targetDensitySum += p;
                diagnosticFlags[i] = denominator < 1e-30 ? (byte)170 : requested < 0 || requested > 1 ? (byte)255 : p is 0 or 1 ? (byte)85 : (byte)0;
                for (int v = 0; v < views; v++)
                {
                    double prediction = Sample(maps[2 * v], m, x, y, size) + delta[v] * p;
                    double target = lo[v] + (hi[v] - lo[v]) * gray;
                    continuousError += settings.Views[v].Weight / weightSum * Math.Pow(prediction - target, 2);
                }
                // Diffuse coverage, even where the hypothetical optical model has no contrast.
                // Model-dependent point placement follows, with local point counts held fixed.
                double adjusted = p + error[(y % 2) * size + x];
                uint phase = unchecked((uint)x * 0x9E3779B1u ^ (uint)y * 0x85EBCA77u ^ settings.Seed);
                phase ^= phase >> 16; phase = unchecked(phase * 0x7FEB352Du); phase ^= phase >> 15;
                int code = adjusted >= .5 + .1 * (phase / 4294967296.0 - .5) ? 1 : 0;
                pixels[i] = (byte)(code * 255);
                double residual = adjusted - code;
                neighbors[0] = (x + dir, y, 7); neighbors[1] = (x - dir, y + 1, 3);
                neighbors[2] = (x, y + 1, 5); neighbors[3] = (x + dir, y + 1, 1);
                int total = 0;
                foreach (var n in neighbors) if (Valid(n.X, n.Y)) total += n.W;
                foreach (var n in neighbors) if (Valid(n.X, n.Y))
                    error[(n.Y % 2) * size + n.X] += residual * n.W / total;
                bool Valid(int xx, int yy) { if (xx < 0 || yy < 0 || xx >= size || yy >= size) return false;
                    double ax = xx * cell - radius, ay = radius - yy * cell; double r = ax * ax + ay * ay; int sourceIndex = ((int)Math.Round(yy * (raster.Height - 1.0) / (size - 1)) * raster.Width
                        + (int)Math.Round(xx * (raster.Width - 1.0) / (size - 1))) * 4;
                    return r >= inner * inner && r <= outer * outer && raster.Pixels[sourceIndex + 3] != 0; }
            }
            Array.Clear(error, (y % 2) * size, size);
        }
        var history = PhysicalRefinement.Refine(pixels, tone, mask, size, cell, maps, m,
            settings, cancellationToken, progress);
        cancellationToken.ThrowIfCancellationRequested();
        Directory.CreateDirectory(cacheDirectory);
        string stage = Path.Combine(cacheDirectory, "." + key + "." + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(stage);
        var result = new DvdPhysicalConversionResult(binary, report, key, size, cell * 1000, (double)clipped / Math.Max(1, active), (double)unobservable / Math.Max(1, active), Path.Combine(directory, "masked.png"));
        try
        {
            int previewSize = Math.Min(512, size);
            var previewCodes = new double[previewSize * previewSize];
            var previewTone = new double[previewCodes.Length];
            var previewCount = new int[previewCodes.Length];
            var previewTotal = new int[previewCodes.Length];
            var diagnostic = new byte[previewCodes.Length];
            for (int y = 0; y < size; y++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                for (int x = 0; x < size; x++)
                {
                    int i = y * size + x;
                    int pi = Math.Min(previewSize - 1, y * previewSize / size) * previewSize + Math.Min(previewSize - 1, x * previewSize / size);
                    previewTotal[pi]++;
                    if (!mask[i]) continue;
                    previewCodes[pi] += pixels[i] / 255.0; previewTone[pi] += tone[i]; previewCount[pi]++;
                    diagnostic[pi] = Math.Max(diagnostic[pi], diagnosticFlags[i]);
                }
            }
            for (int v = 0; v < views; v++)
            {
                double maximum = Math.Max(1e-20, displayMax[v]);
                var predicted = new byte[previewCodes.Length]; var target = new byte[predicted.Length];
                for (int y = 0; y < previewSize; y++) for (int x = 0; x < previewSize; x++)
                {
                    int i = y * previewSize + x; if (previewCount[i] == 0) continue;
                    double r0 = Sample(maps[2 * v], m, x, y, previewSize), d = Sample(maps[2 * v + 1], m, x, y, previewSize) - r0;
                    predicted[i] = DisplayGray((r0 + d * previewCodes[i] / previewCount[i]) / maximum);
                    target[i] = DisplayGray((r0 + d * previewTone[i] / previewCount[i]) / maximum);
                }
                RasterCodec.SavePng(Path.Combine(stage, $"view-{v}.png"), previewSize, previewSize, predicted, true);
                RasterCodec.SavePng(Path.Combine(stage, $"target-{v}.png"), previewSize, previewSize, target, true);
            }
            // Coverage preview is explicitly area averaged, with coverage alpha at
            // disc/mask boundaries. Never let a tiny UI thumbnail point-sample the bits.
            var densityPreview = new byte[previewCodes.Length * 4];
            for (int i = 0; i < previewCodes.Length; i++)
            {
                if (previewCount[i] == 0) continue;
                byte gray = (byte)Math.Round(255 * previewCodes[i] / previewCount[i]);
                densityPreview[4 * i] = densityPreview[4 * i + 1] = densityPreview[4 * i + 2] = gray;
                densityPreview[4 * i + 3] = (byte)Math.Round(255.0 * previewCount[i] / previewTotal[i]);
            }
            RasterCodec.SavePng(Path.Combine(stage, "density.png"), previewSize, previewSize, densityPreview, false);
            RasterCodec.SavePng(Path.Combine(stage, "diagnostic.png"), previewSize, previewSize, diagnostic, true);
            RasterCodec.SavePng(Path.Combine(stage, "binary.png"), size, size, pixels, true);
            byte[] masked = new byte[pixels.Length * 4];
            for (int i = 0; i < pixels.Length; i++)
            {
                masked[i * 4] = masked[i * 4 + 1] = masked[i * 4 + 2] = pixels[i];
                masked[i * 4 + 3] = mask[i] ? (byte)255 : (byte)0;
            }
            RasterCodec.SavePng(Path.Combine(stage, "masked.png"), size, size, masked, false);
            File.WriteAllText(Path.Combine(stage, "report.json"), JsonSerializer.Serialize(new { algorithm = Algorithm,
                settings, sourceHash, request.CanvasRadiusMm, request.InnerRadiusMm, request.OuterRadiusMm, request.TrackPitchNm,
                referenceLow = lo, referenceHigh = hi,
                targetDefinition = "local-mixture: R0 + (R1-R0) * density",
                sourceMeanDensity = sourceDensitySum / Math.Max(1, active),
                targetMeanDensity = targetDensitySum / Math.Max(1, active),
                outputMeanDensity = pixels.Where((_, i) => mask[i]).Sum(b => b / 255.0) / Math.Max(1, active),
                modelOutOfRangeFraction = (double)modelOutOfRange / Math.Max(1, active),
                densityBlockSize = PhysicalRefinement.DensityBlockSize,
                diagnosticLegend = new { endpoint = 85, unobservable = 170, clipped = 255 },
                opticalCheckpoints = maps.Select(map => new[] { map[(m / 2) * m + m / 2], map[(m / 2) * m + 3 * m / 4] }).ToArray(),
                result, globalReferenceFitRms = Math.Sqrt(continuousError / Math.Max(1, active)),
                optimizationAcceptedReduction = history, materialValidated = settings.Provenance == "measured" }, JsonOptions));
            File.WriteAllText(Path.Combine(stage, "result.json"), JsonSerializer.Serialize(result, JsonOptions));
            cancellationToken.ThrowIfCancellationRequested();
            try { Directory.Move(stage, directory); }
            catch (IOException) when (File.Exists(Path.Combine(directory, "result.json"))) { Directory.Delete(stage, true); }
        }
        finally { if (Directory.Exists(stage)) Directory.Delete(stage, true); }
        progress?.Invoke("Complete", 1); return result;
    }

    private static byte DisplayGray(double linear)
    {
        return (byte)Math.Round(255 * EncodeSrgb(linear));
    }

    private static double EncodeSrgb(double linear)
    {
        double value = Math.Clamp(linear, 0, 1);
        return value <= .0031308 ? value * 12.92 : 1.055 * Math.Pow(value, 1 / 2.4) - .055;
    }

    private static double ResampleSource(float[] source, int width, int x, int y, int size)
    {
        if (width == 1) return source[0];
        double scale = (width - 1.0) / (size - 1), fx = x * scale, fy = y * scale;
        if (scale <= 1) return Sample(source, width, x, y, size);
        double left = fx - scale / 2, right = fx + scale / 2, top = fy - scale / 2, bottom = fy + scale / 2;
        double sum = 0, total = 0;
        for (int yy = Math.Max(0, (int)Math.Ceiling(top - .5)); yy <= Math.Min(width - 1, (int)Math.Floor(bottom + .5)); yy++)
            for (int xx = Math.Max(0, (int)Math.Ceiling(left - .5)); xx <= Math.Min(width - 1, (int)Math.Floor(right + .5)); xx++)
            {
                double w = Math.Max(0, Math.Min(right, xx + .5) - Math.Max(left, xx - .5))
                    * Math.Max(0, Math.Min(bottom, yy + .5) - Math.Max(top, yy - .5));
                sum += w * source[yy * width + xx]; total += w;
            }
        return total == 0 ? 1 : sum / total;
    }

    internal static float Sample(float[] map, int m, int x, int y, int size)
    {
        double fx = x * (m - 1.0) / (size - 1), fy = y * (m - 1.0) / (size - 1);
        int a = Math.Min((int)fx, m - 2), b = Math.Min((int)fy, m - 2);
        return (float)((1 - (fy - b)) * ((1 - (fx - a)) * map[b * m + a] + (fx - a) * map[b * m + a + 1])
            + (fy - b) * ((1 - (fx - a)) * map[(b + 1) * m + a] + (fx - a) * map[(b + 1) * m + a + 1]));
    }
}
