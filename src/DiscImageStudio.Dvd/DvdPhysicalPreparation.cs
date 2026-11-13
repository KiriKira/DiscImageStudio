using DiscImageStudio.Core;
using DvdImageSolver.PhysicalImaging;

namespace DiscImageStudio.Dvd;

public static class DvdPhysicalPreparation
{
    public static DvdPhysicalConversionResult Prepare(string source, DvdStreamingOptions options,
        DvdPhysicalConversionSettings settings, string? dataDirectory = null, string volumeLabel = "DISC_IMAGE",
        bool filesAtOuterRing = false, CancellationToken cancellationToken = default,
        Action<string, double>? progress = null)
    {
        options.Validate(); settings.Validate();
        string recipe = options.ReadCompatibility ? options.ReadCompatibilityMode : "off";
        if (settings.Recipe != recipe) throw new ArgumentException("Physical material profile does not match the DVD read-compatibility recipe.");
        var geometry = DvdTrackGeometry.Create(options.InnerRadiusMm, options.OuterRadiusMm,
            options.TotalSectors * 38688d * options.ChannelBitLengthNm * 1e-6,
            options.PitchLinear, options.PitchQuadratic, options.PitchCubic);
        var plan = string.IsNullOrWhiteSpace(dataDirectory) ? null : DvdStreamingGenerator.PrepareHybrid(dataDirectory,
            options, volumeLabel, filesAtOuterRing);
        double bitLength = options.ChannelBitLengthNm * 1e-6;
        double inner = plan is null ? options.InnerRadiusMm : geometry.RadiusAtArcLength(plan.DrawingStartLba * 38688d * bitLength);
        double outer = plan is null ? options.OutputOuterRadiusMm : geometry.RadiusAtArcLength(plan.DrawingEndLbaExclusive * 38688d * bitLength);
        return DvdPhysicalConversion.Prepare(new(source, options.SamplingImageOuterRadiusMm,
            options.InnerRadiusMm, options.OutputOuterRadiusMm, settings, geometry.TrackPitchMm * 1e6, inner, outer, options.PitchLinear, options.PitchQuadratic, options.PitchCubic, options.OuterRadiusMm),
            Path.Combine(Path.GetTempPath(), "DiscImageStudio", "physical-v1"), progress, cancellationToken);
    }
}
