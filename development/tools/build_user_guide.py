"""Build the breeder-facing PDF from checked UI screenshots and plain instructions.

Documentation-only dependency: development/requirements-docs.txt.
Run from the repository root; no app runtime or model is imported.
"""
from pathlib import Path
import shutil
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf/HopBox-Infer-User-Guide.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
FONT=Path('C:/Windows/Fonts')
pdfmetrics.registerFont(TTFont('Guide',str(FONT/'segoeui.ttf')))
pdfmetrics.registerFont(TTFont('GuideBold',str(FONT/'segoeuib.ttf')))
pdfmetrics.registerFontFamily('Guide',normal='Guide',bold='GuideBold',italic='Guide',boldItalic='GuideBold')
W,H=612,792
M=44; CW=W-2*M
INK=colors.HexColor('#153d37');TEAL=colors.HexColor('#267f80')
PALE=colors.HexColor('#eef5ee')
TEXT=colors.HexColor('#273936');MUTED=colors.HexColor('#526660')
LINE=colors.HexColor('#d7e2dc');GOLD=colors.HexColor('#e9b84c')
STYLES={
 'body':ParagraphStyle('body',fontName='Guide',fontSize=11.1,leading=15.6,textColor=TEXT),
 'small':ParagraphStyle('small',fontName='Guide',fontSize=9.2,leading=13,textColor=MUTED),
 'card':ParagraphStyle('card',fontName='Guide',fontSize=10.6,leading=15,textColor=TEXT),
 'h2':ParagraphStyle('h2',fontName='GuideBold',fontSize=14,leading=18,textColor=INK),
}
c=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1,invariant=1)
c.setTitle('HopBox Infer 1.0.0 - Installation and User Guide for Breeders')
c.setAuthor('Collins Wakholi')
c.setSubject('Install, import photos, select compute, preview, process folders and review results')
c.setKeywords('HopBox Infer, hops, breeders, Windows, offline, user guide')
def para(text,x,y,width=CW,style='body',gap=10):
    p=Paragraph(text,STYLES[style]);_,height=p.wrap(width,700)
    if y-height<48:raise RuntimeError('Page overflow: '+text[:70])
    p.drawOn(c,x,y-height)
    return y-height-gap
def base(number,kicker,title,subtitle):
    c.setFillColor(colors.white);c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColor(INK);c.rect(0,H-11,W,11,fill=1,stroke=0)
    c.setFillColor(TEAL);c.setFont('GuideBold',9)
    c.drawString(M,H-42,'HOPBOX INFER  /  USER GUIDE  /  1.0.0')
    c.setFont('Guide',9);c.drawRightString(W-M,H-42,'October 15, 2026')
    c.setFillColor(MUTED);c.setFont('GuideBold',10);c.drawString(M,H-78,kicker.upper())
    c.setFillColor(INK);c.setFont('GuideBold',26);c.drawString(M,H-112,title)
    y=para(subtitle,M,H-129,style='body',gap=18)
    c.setStrokeColor(LINE);c.line(M,38,W-M,38)
    c.setFont('Guide',8);c.setFillColor(MUTED)
    c.drawString(M,25,'Collins Wakholi  |  github.com/collinswakholi')
    c.linkURL('https://github.com/collinswakholi',(M,19,270,34),relative=0)
    c.drawRightString(W-M,25,f'{number} / 8')
    c.bookmarkPage('page'+str(number));c.addOutlineEntry(title,'page'+str(number),level=0)
    return y
def box(text,x,y,width=CW,tone=PALE):
    p=Paragraph(text,STYLES['card']);_,height=p.wrap(width-28,650)
    total=height+25
    if y-total<48:raise RuntimeError('Callout overflow')
    c.setFillColor(tone);c.roundRect(x,y-total,width,total,8,fill=1,stroke=0)
    p.drawOn(c,x+14,y-13-height)
    return y-total-15
def step(number,title,text,y,width=CW):
    c.setFillColor(TEAL);c.circle(M+11,y-10,11,fill=1,stroke=0)
    c.setFillColor(colors.white);c.setFont('GuideBold',10);c.drawCentredString(M+11,y-14,str(number))
    y=para('<b>'+title+'</b><br/>'+text,M+33,y,width-33,gap=13)
    return y
def heading(text,y):return para(text,M,y,style='h2',gap=9)
def screenshot(name,x,y,width,crop=None):
    image=Image.open(ROOT/'docs/images'/name)
    if crop:image=image.crop(crop)
    height=width*image.height/image.width
    if y-height<48:raise RuntimeError('Screenshot overflow')
    c.drawImage(ImageReader(image),x,y-height,width,height,mask='auto')
    return y-height
def bullets(items,y,width=CW,x=M):
    for text in items:y=para('• '+text,x,y,width,gap=8)
    return y
def finish():c.showPage()

y=base(1,'Start here','From photos to measurements',
 'A practical guide for breeders. Follow the same steps for a small sample or a folder of trial photos.')
c.drawImage(str(ROOT/'app/hopbox/static/branding/hopbox.png'),W-M-80,y-80,80,80,mask='auto')
y=para('HopBox Infer finds hops, corrects colour using the chart, reads labels and saves size, shape, colour and texture measurements.',M,y,CW-100)
y=para('<b>Works offline on your PC.</b> No Python, Docker or separate GPU package is needed.',M,y,CW-100,gap=20)
y=heading('Install the Windows app',y)
y=step(1,'Download the installer',
 '<link href="https://github.com/collinswakholi/hopbox-infer/releases/latest" color="#267f80"><u>Open the official release page</u></link> and choose <b>HopBox-Infer-Setup-1.0.0.exe</b>. Do not use GitHub\'s Code &gt; Download ZIP; that is source code.',y)
y=step(2,'Open it and choose a folder',
 'Choose an installation location allowed by your organisation and your shortcut options. A protected folder such as Program Files may need Windows administrator approval.',y)
y=step(3,'Keep the core component and click Install',
 'The standard core already includes CPU and GPU support. You do not need a separate NVIDIA, Intel or AMD pack.',y)
y=step(4,'Open HopBox Infer',
 'Wait for the splash screen and models to load. The interface opens in a browser window, but your photos stay on this computer.',y)
y=box('<b>Before you begin</b><br/>Use a 64-bit Windows PC; Windows 11 has been tested. The installer is about 368 MB and the installed app about 613 MB. Allow extra disk space for installation, photos and results. A GPU is optional.',M,y)
y=para('<b>In this guide:</b> screen map (2), loading photos (3), compute and settings (4), previews (5), batch processing (6), results (7), help (8).',M,y,style='small')
finish()

y=base(2,'Know the screen','Six places to remember',
 'This is the actual HopBox Infer 1.0.0 interface. The example has no photos loaded and uses CPU mode.')
image_y=y;bottom=screenshot('app-overview.png',M,y,CW)
scale=CW/1280
for number,px,py in [(1,235,111),(2,231,275),(3,872,33),(4,1230,136),(5,1167,505),(6,1229,625)]:
    x=M+px*scale;cy=image_y-py*scale
    c.setFillColor(GOLD);c.circle(x,cy,9,fill=1,stroke=0)
    c.setFillColor(INK);c.setFont('GuideBold',9);c.drawCentredString(x,cy-3,str(number))
y=bottom-23
for title,text in [
 ('1  Import folder','Add photos from your computer. Tick folders to include them in a run.'),
 ('2  Folders and Photos','Click a folder name, then a photo name to inspect an image.'),
 ('3  Preview buttons','Chart, Corrected and Detections help check a photo before a full run.'),
 ('4  Run settings','Choose compute, batch size, colour settings, label reading and output folder. Scroll inside this panel for more settings.'),
 ('5  Run folders / Stop','Start processing the ticked folders, or stop the current job.'),
 ('6  Progress and Activity','Follow progress, completion messages and any problems.')]:
    y=para('<b>'+title+'</b>  '+text,M,y,gap=10)
y=para('The screen layout may vary with window size. You can drag the panel dividers to give the photo or settings more room.',M,y,style='small')
finish()

y=base(3,'Load images','Bring in a trial folder',
 'Keep the original photographs together. Use the established HopBox camera, tray, chart and label setup.')
y=step(1,'Click Import folder',
 'Choose the folder containing your photographs. You can also choose a parent folder that contains several trial folders.',y)
y=step(2,'Check the folder boxes',
 'Tick only the folders you want to process. Ticking a parent includes its children. The selected-photo count helps check the intended scope.',y)
y=step(3,'Choose a photo to inspect',
 'Click a folder name in Folders. Then click a photo name in Photos. Clicking a photo for preview does not limit Run folders to that one photo.',y)
y=box('<b>Example folder arrangement</b><br/><font face="GuideBold">Hop photos<br/>&nbsp;&nbsp;Trial A<br/>&nbsp;&nbsp;&nbsp;&nbsp;IMG_0001.JPG<br/>&nbsp;&nbsp;&nbsp;&nbsp;IMG_0002.JPG<br/>&nbsp;&nbsp;Trial B<br/>&nbsp;&nbsp;&nbsp;&nbsp;IMG_0003.JPG</font><br/>Import Hop photos to see both trials, then tick the trials you need.',M,y)
y=step(4,'Choose an Output folder before running',
 'In Run settings, scroll down and click Browse beside Output folder. Choose a new results folder outside the photo folders. Importing a folder supplies a suggested output location; you can change it.',y)
y=box('<b>Good inputs matter</b><br/>Keep hops, the colour chart and the sample label sharp and visible. Avoid overlaps, deep shadows, glare and cut-off chart patches. Review representative easy and difficult photos before processing a trial.',M,y)
y=para('Supported: JPG, JPEG, PNG, TIF, TIFF and BMP. RAW camera files are not listed. Folders beginning with a dot or underscore are skipped. Clear images removes the imported list, not your original photos.',M,y,style='small')
finish()

y=base(4,'Choose settings','CPU or GPU?',
 'Compute device chooses your main processor (CPU) or graphics processor (GPU) for detection. Keep settings consistent within a study.')
bottom=screenshot('app-overview.png',M,y,175,crop=(905,80,1275,408))
x=M+194;rw=CW-194
y=para('<b>Auto</b><br/>A good starting point. The app chooses from available devices and normally prefers a dedicated GPU.',x,y,rw)
y=para('<b>CPU</b><br/>Works without a supported GPU. Use it if the GPU cannot start or processing is unreliable. It can take longer.',x,y,rw)
y=para('<b>A named GPU</b><br/>Choose the NVIDIA or Intel device listed on your PC. A GPU can speed up finding objects; total speed still depends on the photos and other processing.',x,y,rw)
y=min(y,bottom)-16
y=box('<b>AMD Radeon</b><br/>The app uses DirectML for Windows GPUs, but physical Radeon execution has not been validated. A shared NVIDIA VM test does not prove AMD compatibility. CPU remains the fallback.',M,y)
y=heading('Start with the supplied settings',y)
for label,value in [('Batch size','1 photo'),('Image size / Confidence %','2048 / 58'),('Color method','CCM 3x3'),('Gamma correction / White balance','Both on'),('Read labels','QR code, then printed text'),('Save hop outlines','On')]:
    c.setStrokeColor(LINE);c.line(M,y+2,W-M,y+2)
    y1=para('<b>'+label+'</b>',M,y-6,235,style='card',gap=8)
    y2=para(value,M+250,y-6,CW-250,style='card',gap=8)
    y=min(y1,y2)-3
y=para('Use your project\'s agreed settings if they differ. Small numerical differences can occur between CPU and GPU; record the version, device and settings used. Load model is an advanced option and is not needed for the supplied hop model.',M,y-7,style='small')
finish()

y=base(5,'Preview first','Check three views',
 'Select a photo, then click a preview button at the top. Wait for the preview to finish before changing settings.')
bottom=screenshot('app-overview.png',M,y,CW,crop=(535,1,919,70));y=bottom-22
for number,title,text in [
 (1,'Detections - were the hops found?',
  'Inspect the outlines. Look for missed hops, background objects included as hops, and two overlapping hops merged into one. Check a few different trays, not only the clearest photo.'),
 (2,'Chart - was the colour chart found?',
  'Check that the detected chart and patches belong to the chart in your photo. A blurred, cut-off or poorly lit chart can prevent correction and may leave the photo without measured rows.'),
 (3,'Corrected - does the image look reasonable?',
  'Inspect the corrected colours alongside your expected tray appearance. Colour plots becomes available after a suitable corrected preview; it is optional for routine use.')]:
    y=step(number,title,text,y)
y=box('<b>Read labels</b><br/>The default tries the QR code first, then printed text when needed. Check the saved label information against the sample label before analysis. Blurred or rotated text can be misread, especially digits and identifiers.',M,y)
y=heading('A preview is a quality check',y)
y=para('A single-photo preview does not create the full Features.csv measurement table. Once the previews look acceptable, use Run folders to produce the full results for the ticked folders.',M,y)
y=para('If results look wrong, check framing, focus, lighting and chart visibility first. Do not lower confidence or change correction settings midway through a study without recording and validating the change.',M,y)
y=para('The guide uses an empty app screen so no private trial photographs or sample identifiers appear in these screenshots.',M,y,style='small')
finish()

y=base(6,'Process a batch','Run folders, not just one photo',
 'Folder selection controls which photographs are processed. Batch size controls how many photos are handled together.')
y=step(1,'Tick the folders for this run',
 'You may select one trial or several. Check the selected-photo total. The highlighted photo is for viewing; the ticked folders determine the run.',y)
y=step(2,'Check the Output folder',
 'Choose a fresh, writable results location. Existing results with matching folder and file names can be replaced by another run.',y)
y=step(3,'Choose Batch size',
 '<b>1 photo</b> is the starting point and still processes all photos in the selected folders. Try 2 next; 4 and 8 need more memory. Larger is not always faster.',y)
y=step(4,'Click Run folders and wait',
 'Watch Progress and Activity. The app finds hops, locates the chart, corrects colour, reads labels and calculates measurements. Some steps can take longer than others.',y)
y=step(5,'Confirm completion',
 'Look for the final completion message and the Features.csv output messages. Review any chart, label or detection problems before accepting the run.',y)
y=box('<b>Keep the app open during processing</b><br/>Use Stop if you need to cancel and wait for it to stop. Exit ends the app. A cancelled or interrupted run may leave partial images or tables; do not treat those as a complete trial. Rerun into a new output folder.',M,y)
y=heading('Two different kinds of batch',y)
y=para('<b>Several folders:</b> tick multiple trials and click Run folders once.<br/><b>Several photos at once:</b> increase Batch size only when there is enough memory.<br/>The app may apply memory safeguards; a requested batch is not a promise that every processing stage uses that many photos at once.',M,y)
finish()

y=base(7,'Use the results','Find the table and images',
 'Open your chosen Output folder after the run finishes. Each processed photo folder has its own results.')
left_y=y
y=box('<b>Example output</b><br/>Your chosen results folder<br/>&nbsp;&nbsp;Trial A<br/>&nbsp;&nbsp;&nbsp;&nbsp;<b>Features.csv</b><br/>&nbsp;&nbsp;&nbsp;&nbsp;<b>Image_Predictions</b><br/>&nbsp;&nbsp;&nbsp;&nbsp;<b>Chart_Detections</b><br/>&nbsp;&nbsp;Trial B<br/>&nbsp;&nbsp;&nbsp;&nbsp;Features.csv<br/>&nbsp;&nbsp;&nbsp;&nbsp;...',M,y,265)
bottom=screenshot('output-settings.png',M+281,left_y,CW-281,crop=(905,193,1275,432))
y=min(y,bottom)-4
y=para('<b>Features.csv</b><br/>Open this table in Excel or your analysis program. Each row describes a measured hop and includes its source photo, label information and calculated traits. A photo can contribute several rows.',M,y)
y=para('<b>Image_Predictions</b><br/>Corrected photos with hop outlines when Save hop outlines is enabled. Use these to review segmentation and associate rows with visible hops.',M,y)
y=para('<b>Chart_Detections</b><br/>Photos showing the detected colour chart. Review these when a correction looks unusual or a photo contributes no measurements.',M,y)
y=box('<b>Before combining trials</b><br/>Check the number of processed photos, missed/merged hops, chart detection and label text. A photo without a detected chart or measured hops may contribute no rows. Check Activity as well as the CSV.',M,y)
y=para('Excel may change long numbers or remove leading zeros when opening CSVs. For important sample identifiers, use Data &gt; From Text/CSV and keep the QR_info/label column as text. Keep an untouched copy of the original CSV.',M,y,style='small')
y=para('Original input photographs are not modified. Keep the original photos, CSVs and your run settings together as the study record.',M,y,style='small')
finish()

y=base(8,'Help and good habits','If something does not work',
 'Start with a small folder and Batch size 1. Check Activity for the message that explains the problem.')
for title,text in [
 ('No photos in the list','Choose the actual photo folder and check the supported file types on page 3. Folders beginning with a dot or underscore are skipped.'),
 ('Slow processing or not enough memory','Reduce Batch size to 1, close other applications, and try CPU. Allow the current job to finish or stop before changing compute.'),
 ('Missing hops, chart or label','Check Detections and Chart on representative photos. Improve focus, lighting, separation and framing. Verify printed text and identifiers by eye.'),
 ('App will not open or files appear missing','Use the complete installer or keep the entire portable folder together. The release includes SECURITY_AND_STARTUP.md for diagnosis. If security software blocks the app, follow your organisation\'s approved process; do not disable protection.'),
 ('A run was stopped or the app was closed','Treat saved output as incomplete. Start the app again and rerun into a fresh output folder. Wait for completion before using Exit.')]:
    y=para('<b>'+title+'</b><br/>'+text,M,y,gap=13)
y=box('<b>Keep comparisons consistent</b><br/>Record version 1.0.0, compute device, batch size, image size, confidence and colour settings. Check measurements before combining this Windows version with results from the older Docker/PyTorch version.',M,y)
y=para('<b>Downloads and help</b><br/><link href="https://github.com/collinswakholi/hopbox-infer" color="#267f80"><u>github.com/collinswakholi/hopbox-infer</u></link><br/>When reporting a problem, include the app version, compute choice and Activity message. Review logs and photos before posting; they may contain private sample information.',M,y,style='small')
finish()
c.save()
destination=ROOT/'docs/HopBox-Infer-User-Guide.pdf'
shutil.copyfile(OUT,destination)
print('Created 8-page user guide:',destination)
