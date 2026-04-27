param(
    [string]$OutDir = "D:\rz\danzi\essvlm\figures"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

$ScriptsDir = Join-Path $OutDir "scripts"
$CandidatesDir = Join-Path $OutDir "candidates"
New-Item -ItemType Directory -Force -Path $OutDir, $ScriptsDir, $CandidatesDir | Out-Null

$Palette = @{
    Bg = [System.Drawing.Color]::FromArgb(248, 250, 252)
    White = [System.Drawing.Color]::White
    Ink = [System.Drawing.Color]::FromArgb(17, 24, 39)
    Muted = [System.Drawing.Color]::FromArgb(75, 85, 99)
    Light = [System.Drawing.Color]::FromArgb(229, 231, 235)
    Teacher = [System.Drawing.Color]::FromArgb(31, 157, 154)
    TeacherLight = [System.Drawing.Color]::FromArgb(210, 245, 242)
    Student = [System.Drawing.Color]::FromArgb(245, 158, 11)
    StudentLight = [System.Drawing.Color]::FromArgb(255, 237, 213)
    Loss = [System.Drawing.Color]::FromArgb(220, 38, 38)
    LossLight = [System.Drawing.Color]::FromArgb(254, 226, 226)
    Navy = [System.Drawing.Color]::FromArgb(15, 23, 42)
    Green = [System.Drawing.Color]::FromArgb(22, 163, 74)
    Blue = [System.Drawing.Color]::FromArgb(37, 99, 235)
}

function New-Figure {
    param([int]$Width, [int]$Height)
    $bmp = New-Object System.Drawing.Bitmap($Width, $Height)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    $g.Clear($Palette.Bg)
    return @($bmp, $g)
}

function Save-Figure {
    param($Bitmap, $Graphics, [string]$Path)
    $Graphics.Dispose()
    $Bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
    $Bitmap.Dispose()
}

function New-Font {
    param([float]$Size, [string]$Style = "Regular")
    $fontStyle = [System.Drawing.FontStyle]::$Style
    return New-Object System.Drawing.Font("Segoe UI", $Size, $fontStyle)
}

function Draw-Text {
    param($G, [string]$Text, [float]$X, [float]$Y, [float]$W, [float]$H, [float]$Size = 18, [string]$Style = "Regular", $Color = $Palette.Ink, [string]$Align = "Center")
    $font = New-Font $Size $Style
    $brush = New-Object System.Drawing.SolidBrush($Color)
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = [System.Drawing.StringAlignment]::$Align
    $format.LineAlignment = [System.Drawing.StringAlignment]::Center
    $format.FormatFlags = 0
    $rect = New-Object System.Drawing.RectangleF($X, $Y, $W, $H)
    $G.DrawString($Text, $font, $brush, $rect, $format)
    $font.Dispose(); $brush.Dispose(); $format.Dispose()
}

function Draw-RoundRect {
    param($G, [float]$X, [float]$Y, [float]$W, [float]$H, [float]$Radius, $FillColor, $BorderColor = $Palette.Light, [float]$BorderWidth = 2)
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = $Radius * 2
    $path.AddArc($X, $Y, $d, $d, 180, 90)
    $path.AddArc($X + $W - $d, $Y, $d, $d, 270, 90)
    $path.AddArc($X + $W - $d, $Y + $H - $d, $d, $d, 0, 90)
    $path.AddArc($X, $Y + $H - $d, $d, $d, 90, 90)
    $path.CloseFigure()
    $brush = New-Object System.Drawing.SolidBrush($FillColor)
    $pen = New-Object System.Drawing.Pen($BorderColor, $BorderWidth)
    $G.FillPath($brush, $path)
    $G.DrawPath($pen, $path)
    $brush.Dispose(); $pen.Dispose(); $path.Dispose()
}

function Draw-Arrow {
    param($G, [float]$X1, [float]$Y1, [float]$X2, [float]$Y2, $Color = $Palette.Ink, [float]$Width = 3)
    $pen = New-Object System.Drawing.Pen($Color, $Width)
    $cap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(6, 7, $true)
    $pen.CustomEndCap = $cap
    $G.DrawLine($pen, $X1, $Y1, $X2, $Y2)
    $cap.Dispose(); $pen.Dispose()
}

function Draw-Badge {
    param($G, [string]$Text, [float]$X, [float]$Y, [float]$W, [float]$H, $FillColor, $TextColor = $Palette.White)
    Draw-RoundRect $G $X $Y $W $H 18 $FillColor $FillColor 1
    Draw-Text $G $Text $X $Y $W $H 17 "Bold" $TextColor
}

function Draw-Box {
    param($G, [string]$Title, [string]$Sub, [float]$X, [float]$Y, [float]$W, [float]$H, $Fill, $Border)
    Draw-RoundRect $G $X $Y $W $H 22 $Fill $Border 3
    Draw-Text $G $Title ($X+14) ($Y+12) ($W-28) 34 17 "Bold" $Palette.Ink
    if ($Sub -ne "") {
        Draw-Text $G $Sub ($X+18) ($Y+48) ($W-36) ($H-56) 14 "Regular" $Palette.Muted
    }
}

function Make-HeroOverview {
    $fig = New-Figure 2000 760; $bmp = $fig[0]; $g = $fig[1]
    Draw-Text $g "ESS-VLM-EventSeg" 50 30 600 50 32 "Bold" $Palette.Navy "Near"
    Draw-Text $g "Offline FC-CLIP supervision + Top-2 Soft distillation for event semantic segmentation" 52 84 1200 36 18 "Regular" $Palette.Muted "Near"
    Draw-Badge $g "best checkpoint mIoU 53.05 -> 56.81" 1420 45 460 48 $Palette.Navy

    Draw-Box $g "DDD17 Data" "paired frames + events" 70 260 270 120 $Palette.White $Palette.Light
    Draw-Box $g "FC-CLIP Offline Teacher" "frames -> semantic artifacts" 500 155 360 120 $Palette.TeacherLight $Palette.Teacher
    Draw-Box $g "ESS Event Student" "events -> segmentation training" 500 445 360 120 $Palette.StudentLight $Palette.Student
    Draw-Box $g "Hard Pseudolabels" "top-1 dense masks" 1010 115 300 105 $Palette.White $Palette.Teacher
    Draw-Box $g "Top-2 Soft Labels" "top-1/top-2 probabilities" 1010 280 300 105 $Palette.White $Palette.Teacher
    Draw-Badge $g "Hard CE" 1390 135 170 52 $Palette.Loss
    Draw-Badge $g "Top-2 Soft Distillation" 1390 302 285 52 $Palette.Loss
    Draw-Box $g "Event Seg. Output" "final prediction from ESS events" 1585 470 330 120 $Palette.White $Palette.Navy

    Draw-Arrow $g 340 285 500 215 $Palette.Teacher 4
    Draw-Arrow $g 340 350 500 505 $Palette.Student 4
    Draw-Arrow $g 860 215 1010 168 $Palette.Teacher 4
    Draw-Arrow $g 860 215 1010 333 $Palette.Teacher 4
    Draw-Arrow $g 1310 168 1390 161 $Palette.Loss 3
    Draw-Arrow $g 1310 333 1390 328 $Palette.Loss 3
    Draw-Arrow $g 1475 354 860 475 $Palette.Loss 2
    Draw-Arrow $g 1560 161 860 455 $Palette.Loss 2
    Draw-Arrow $g 860 505 1585 530 $Palette.Student 4
    Draw-Text $g "FC-CLIP is used offline to generate supervision artifacts; it is not in the online training forward path." 250 680 1500 34 17 "Regular" $Palette.Muted
    Save-Figure $bmp $g (Join-Path $OutDir "hero_overview.png")
}

function Make-MethodPipeline {
    $fig = New-Figure 1600 1000; $bmp = $fig[0]; $g = $fig[1]
    Draw-Text $g "Teacher-to-Student Pipeline" 60 30 700 52 31 "Bold" $Palette.Navy "Near"
    Draw-Text $g "DDD17 frames are processed once by FC-CLIP; ESS trains on events with hard and soft supervision." 64 82 1100 34 17 "Regular" $Palette.Muted "Near"

    Draw-Box $g "DDD17 Frame" "paired image side" 90 190 230 105 $Palette.White $Palette.Light
    Draw-Box $g "DDD17 Events" "event tensor side" 90 620 230 105 $Palette.White $Palette.Light
    Draw-Box $g "FC-CLIP Offline Teacher" "dense semantic prediction" 420 180 330 125 $Palette.TeacherLight $Palette.Teacher
    Draw-Box $g "Offline Artifacts" "hard masks + top-2 ids/probabilities" 860 180 350 125 $Palette.White $Palette.Teacher
    Draw-Box $g "ESS Event Student" "event semantic segmentation network" 500 610 350 130 $Palette.StudentLight $Palette.Student
    Draw-Box $g "Segmentation Output" "student prediction" 1030 610 360 130 $Palette.White $Palette.Navy
    Draw-Badge $g "Hard CE" 930 390 170 54 $Palette.Loss
    Draw-Badge $g "Top-2 Soft Distillation CE" 900 470 350 54 $Palette.Loss

    Draw-Arrow $g 320 242 420 242 $Palette.Teacher 4
    Draw-Arrow $g 750 242 860 242 $Palette.Teacher 4
    Draw-Arrow $g 320 672 500 672 $Palette.Student 4
    Draw-Arrow $g 850 675 1030 675 $Palette.Student 4
    Draw-Arrow $g 1030 420 760 610 $Palette.Loss 3
    Draw-Arrow $g 1070 498 780 610 $Palette.Loss 3

    Draw-Text $g "Precompute stage" 470 330 260 34 18 "Bold" $Palette.Teacher
    Draw-Text $g "ESS training stage" 520 770 300 34 18 "Bold" $Palette.Student
    Draw-Text $g "No online teacher inference during ESS training" 520 815 610 30 17 "Regular" $Palette.Muted
    Save-Figure $bmp $g (Join-Path $OutDir "method_pipeline.png")
}

function Make-Top2Explainer {
    $fig = New-Figure 1500 850; $bmp = $fig[0]; $g = $fig[1]
    Draw-Text $g "Top-2 Soft Distillation Explainer" 55 32 800 50 30 "Bold" $Palette.Navy "Near"
    Draw-Text $g "Illustrative schematic: hard labels keep only the answer; Top-2 Soft keeps uncertainty." 58 86 1100 34 18 "Regular" $Palette.Muted "Near"

    Draw-Box $g "Teacher pixel distribution" "" 80 180 400 520 $Palette.TeacherLight $Palette.Teacher
    $labels = @("car", "pole", "road", "other")
    $vals = @(0.70, 0.20, 0.05, 0.05)
    for ($i=0; $i -lt $labels.Count; $i++) {
        $y = 290 + $i*72
        Draw-Text $g $labels[$i] 130 $y 80 30 18 "Bold" $Palette.Ink "Near"
        Draw-RoundRect $g 230 $y 190 28 10 $Palette.White $Palette.White 1
        Draw-RoundRect $g 230 $y (190*$vals[$i]) 28 10 $Palette.Teacher $Palette.Teacher 1
        Draw-Text $g ("{0:N2}" -f $vals[$i]) 425 $y 55 28 15 "Regular" $Palette.Muted "Near"
    }

    Draw-Box $g "Hard label" "one-hot target" 590 230 300 360 $Palette.White $Palette.Light
    Draw-Badge $g "car = 1.00" 650 360 180 60 $Palette.Navy
    Draw-Text $g "Only the strongest class remains." 625 455 230 60 16 "Regular" $Palette.Muted

    Draw-Box $g "Top-2 soft label" "" 1010 230 360 360 $Palette.White $Palette.Teacher
    Draw-Badge $g "car = 0.78" 1080 330 210 54 $Palette.Teacher
    Draw-Badge $g "pole = 0.22" 1080 410 210 54 $Palette.Student
    Draw-Text $g "Student learns the answer and the ambiguity." 1055 515 270 50 16 "Regular" $Palette.Muted

    Draw-Text $g "Illustrative values only" 118 625 320 34 15 "Regular" $Palette.Muted
    Draw-Arrow $g 480 382 590 382 $Palette.Navy 3
    Draw-Arrow $g 480 560 1010 500 $Palette.Teacher 3
    Save-Figure $bmp $g (Join-Path $OutDir "top2_uncertainty_explainer.png")
}

function Make-MiouComparison {
    $fig = New-Figure 1200 780; $bmp = $fig[0]; $g = $fig[1]
    Draw-Text $g "Validation mIoU on DDD17" 60 35 700 50 30 "Bold" $Palette.Navy "Near"
    Draw-Text $g "Top-2 Soft value is the best validation checkpoint." 62 86 730 32 17 "Regular" $Palette.Muted "Near"

    $names = @("ESS baseline", "ESS + FC-CLIP", "+ Top-2 Soft`nbest ckpt")
    $vals = @(53.05, 56.36, 56.8124)
    $colors = @($Palette.Muted, $Palette.Teacher, $Palette.Student)
    $base = 50.0; $max = 57.5
    $x0 = 150; $barW = 190; $gap = 120; $plotH = 500; $yBase = 650
    for ($i=0; $i -lt 3; $i++) {
        $h = (($vals[$i] - $base) / ($max - $base)) * $plotH
        $x = $x0 + $i*($barW+$gap)
        $y = $yBase - $h
        Draw-RoundRect $g $x $y $barW $h 18 $colors[$i] $colors[$i] 1
        Draw-Text $g ("{0:N4}" -f $vals[$i]) ($x-15) ($y-42) ($barW+30) 32 18 "Bold" $Palette.Ink
        Draw-Text $g $names[$i] ($x-45) ($yBase+20) ($barW+90) 70 16 "Regular" $Palette.Ink
    }
    Draw-Text $g "mIoU" 55 300 70 30 16 "Bold" $Palette.Muted
    Draw-Text $g "baseline -> best ckpt`n+3.7624 mIoU" 850 275 230 58 16 "Bold" $Palette.Green
    Save-Figure $bmp $g (Join-Path $OutDir "miou_comparison.png")
}

function Make-CheckpointCurve {
    $fig = New-Figure 1350 820; $bmp = $fig[0]; $g = $fig[1]
    Draw-Text $g "Top-2 Soft Checkpoint Selection" 60 35 760 50 30 "Bold" $Palette.Navy "Near"
    Draw-Text $g "Epoch 15 is the best validation checkpoint; Epoch 18 is the final epoch." 62 86 930 32 17 "Regular" $Palette.Muted "Near"
    $epochs = @(13,14,15,16,17,18)
    $miou = @(52.0516,53.0631,56.8124,54.3424,55.7079,56.1059)
    $left=120; $top=150; $w=1060; $h=520; $minY=51.5; $maxY=57.2
    $penAxis = New-Object System.Drawing.Pen($Palette.Light, 2)
    $g.DrawLine($penAxis, $left, $top+$h, $left+$w, $top+$h)
    $g.DrawLine($penAxis, $left, $top, $left, $top+$h)
    $penAxis.Dispose()
    function YMap($v) { return $top + ($maxY - $v)/($maxY-$minY)*$h }
    function XMap($idx) { return $left + $idx/5*$w }
    $refs = @(@(56.7029, "previous top-2 best 56.7029", $Palette.Loss), @(56.36, "ESS + FC-CLIP 56.36", $Palette.Teacher))
    foreach ($r in $refs) {
        $y = YMap $r[0]
        $pen = New-Object System.Drawing.Pen($r[2], 2)
        $pen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash
        $g.DrawLine($pen, $left, $y, $left+$w, $y)
        $pen.Dispose()
        Draw-Text $g $r[1] ($left+$w-330) ($y-28) 320 24 14 "Regular" $r[2] "Far"
    }
    $linePen = New-Object System.Drawing.Pen($Palette.Student, 4)
    for ($i=0; $i -lt 5; $i++) { $g.DrawLine($linePen, (XMap $i), (YMap $miou[$i]), (XMap ($i+1)), (YMap $miou[$i+1])) }
    $linePen.Dispose()
    for ($i=0; $i -lt 6; $i++) {
        $x = XMap $i; $y = YMap $miou[$i]
        $brush = New-Object System.Drawing.SolidBrush($(if ($epochs[$i] -eq 15) { $Palette.Loss } else { $Palette.Student }))
        $g.FillEllipse($brush, $x-8, $y-8, 16, 16)
        $brush.Dispose()
        Draw-Text $g ([string]$epochs[$i]) ($x-25) ($top+$h+18) 50 24 14 "Regular" $Palette.Muted
    }
    Draw-Badge $g "Epoch 15: 56.8124" ((XMap 2)-95) ((YMap 56.8124)-75) 210 45 $Palette.Loss
    Draw-Text $g "Epoch" ($left+$w/2-40) ($top+$h+55) 80 28 15 "Bold" $Palette.Muted
    Draw-Text $g "Val. mIoU" 18 ($top+220) 90 30 15 "Bold" $Palette.Muted
    Draw-Text $g "Best validation checkpoint, not final epoch" 780 725 430 34 18 "Bold" $Palette.Navy
    Save-Figure $bmp $g (Join-Path $OutDir "checkpoint_curve.png")
}

function Make-ContributionWaterfall {
    $fig = New-Figure 1300 800; $bmp = $fig[0]; $g = $fig[1]
    Draw-Text $g "Contribution Decomposition" 60 35 700 50 30 "Bold" $Palette.Navy "Near"
    Draw-Text $g "mIoU improvement comes from VLM adaptation and Top-2 Soft best-checkpoint optimization." 62 86 1050 32 17 "Regular" $Palette.Muted "Near"
    $base=53.05; $fc=3.31; $top2=0.4524; $final=56.8124; $min=52.5; $max=57.2
    $left=140; $yBase=650; $plotH=470; $barW=190; $gap=85
    function HMap($v) { return ($v-$min)/($max-$min)*$plotH }
    $items = @(
        @("ESS baseline", $base, $Palette.Muted, $min),
        @("+ FC-CLIP`npseudolabel", $fc, $Palette.Teacher, $base),
        @("+ Top-2 Soft`nbest ckpt", $top2, $Palette.Student, $base+$fc),
        @("Final best", $final, $Palette.Green, $min)
    )
    for ($i=0; $i -lt $items.Count; $i++) {
        $x=$left+$i*($barW+$gap)
        if ($i -eq 0 -or $i -eq 3) {
            $h=HMap $items[$i][1]; $y=$yBase-$h
            Draw-RoundRect $g $x $y $barW $h 16 $items[$i][2] $items[$i][2] 1
            Draw-Text $g ("{0:N4}" -f $items[$i][1]) ($x-10) ($y-38) ($barW+20) 28 17 "Bold" $Palette.Ink
        } else {
            $start=HMap $items[$i][3]; $h=HMap($items[$i][3]+$items[$i][1])-$start; $y=$yBase-$start-$h
            Draw-RoundRect $g $x $y $barW $h 16 $items[$i][2] $items[$i][2] 1
            Draw-Text $g ("+{0:N4}" -f $items[$i][1]) ($x-10) ($y-38) ($barW+20) 28 17 "Bold" $Palette.Ink
        }
        Draw-Text $g $items[$i][0] ($x-30) ($yBase+20) ($barW+60) 60 15 "Regular" $Palette.Ink
    }
    Draw-Badge $g "VPR explored: 54.45, not selected" 800 118 390 46 $Palette.Navy
    Draw-Text $g "Val. mIoU" 55 350 80 40 15 "Bold" $Palette.Muted
    Save-Figure $bmp $g (Join-Path $OutDir "contribution_waterfall.png")
}

function Make-Readme {
    $content = @'
# Figure Guide for ESS-VLM-EventSeg

This document explains the README figures for the ESS-VLM-EventSeg GitHub page.

Important metric note: `56.8124` is the best validation checkpoint result at Epoch 15, not the final epoch result. The final epoch result is `56.1059` at Epoch 18.

FC-CLIP should be described as an offline visual-language teacher used to generate supervision artifacts. Do not describe or imply FC-CLIP online inference during ESS training. Do not claim SOTA.

## Required Figures

| Figure | Type | Purpose | Recommended README Position |
| --- | --- | --- | --- |
| `hero_overview.png` | Schematic overview with real metric summary | Gives a 5-10 second overview: DDD17 events go to the ESS student, frames are processed by the FC-CLIP offline teacher, and training uses hard pseudolabel CE plus Top-2 Soft distillation. The result badge says `mIoU 53.05 -> 56.81` and `best checkpoint`. | Top of README, before the project introduction |
| `method_pipeline.png` | Schematic method diagram | Explains the full teacher-student pipeline: DDD17 paired frames/events, FC-CLIP offline teacher outputs, hard pseudolabels, top-2 ids/probabilities, ESS event student, hard CE, and Top-2 Soft distillation CE. | Method section |
| `top2_uncertainty_explainer.png` | Explanatory schematic, not measured pixel data | Explains why Top-2 Soft labels contain richer supervision than one-hot hard labels. The example distribution is illustrative only. | Top-2 Soft distillation subsection |
| `contribution_waterfall.png` | Real metric-backed figure | Shows the contribution path from `ESS baseline: 53.05` to `+ FC-CLIP pseudolabel: +3.31`, then `+ Top-2 Soft best checkpoint: +0.4524`, ending at `56.8124`. VPR is annotated as explored but not selected. | Contribution or ablation summary section |
| `checkpoint_curve.png` | Real metric-backed figure | Shows validation mIoU from Epoch 13 to Epoch 18 and highlights `Epoch 15: 56.8124` as the best validation checkpoint. | Experiment notes or checkpoint selection section |
| `miou_comparison.png` | Real metric-backed figure | Provides the simplest metric comparison: `ESS baseline: 53.05`, `ESS + FC-CLIP: 56.36`, and `ESS + FC-CLIP + Top-2 Soft: 56.8124 best checkpoint`. | Results section, preferably after the result table |

## Suggested README Order

1. `hero_overview.png`
2. Short project introduction
3. Result table
4. `miou_comparison.png`
5. `method_pipeline.png`
6. `top2_uncertainty_explainer.png`
7. `checkpoint_curve.png`
8. `contribution_waterfall.png`

## Metric Facts Used by Result Figures

| Item | mIoU | Acc | Note |
| --- | ---: | ---: | --- |
| ESS official baseline | 53.05 | 87.01 | DDD17 official ESS UDA validation |
| ESS + FC-CLIP dense pseudolabel | 56.36 | 89.82 | Stable VLM pseudolabel branch |
| VPR exploration | 54.45 | 89.60 | Explored, not selected as final method |
| Previous Top-2 Soft best | 56.7029 | 89.5777 | Previous best checkpoint |
| Current Top-2 Soft w0015 best | 56.8124 | 89.5776 | Epoch 15, best validation checkpoint |
| Current Top-2 Soft w0015 final | 56.1059 | 89.6507 | Epoch 18, final epoch |

## Wording Constraints

Use these descriptions consistently:

- `FC-CLIP offline teacher`
- `best validation checkpoint`
- `Top-2 Soft distillation`
- `Validation mIoU on DDD17`

Avoid these unsupported claims:

- Do not call the result SOTA.
- Do not describe FC-CLIP as running online during ESS training.
- Do not present the Top-2 uncertainty example as a real measured pixel distribution.
- Do not present VPR as part of the final improvement path.
'@
    Set-Content -LiteralPath (Join-Path $OutDir "README_figures.md") -Value $content -Encoding UTF8
}

Make-HeroOverview
Make-MethodPipeline
Make-Top2Explainer
Make-MiouComparison
Make-CheckpointCurve
Make-ContributionWaterfall
Make-Readme

Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $ScriptsDir "make_essvlm_figures.ps1") -Force
Get-ChildItem -LiteralPath $OutDir -Filter *.png | Select-Object Name, Length, LastWriteTime
