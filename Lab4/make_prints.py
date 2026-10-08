"""Export a matched 40 cm paper pair as four Letter pages per patch."""
import argparse
import io
from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from common import ROOT


def make_pdf(image_path,output,label):
    with Image.open(image_path) as im: image=im.convert('RGB').resize((2048,2048),Image.Resampling.BILINEAR)
    pdf=canvas.Canvas(str(output),pagesize=letter,pageCompression=1)
    pdf.setTitle('Lab 4 '+label+' patch 40 cm')
    locations=['Top left','Top right','Bottom left','Bottom right']
    for index,location in enumerate(locations):
        x,y=index%2,index//2
        tile=image.crop((x*1024,y*1024,(x+1)*1024,(y+1)*1024))
        pdf.setFont('Helvetica-Bold',13);pdf.drawString(30,757,'Lab 4 | '+label+' | '+location)
        pdf.setFont('Helvetica',10);pdf.drawString(30,739,'Print in color at 100% / actual size. Each square must measure 20 cm.')
        pdf.drawImage(ImageReader(tile),(letter[0]-20*cm)/2,4.4*cm,width=20*cm,height=20*cm)
        pdf.drawString(30,95,'Trim at the square edges. Join all four tiles from the back to make 40 x 40 cm.')
        pdf.drawString(30,78,'Top row: top left + top right. Bottom row: bottom left + bottom right.')
        pdf.drawString(30,61,'Hold the assembled patch flat against the torso, facing the camera.')
        pdf.showPage()
    pdf.save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patch',type=Path,default=ROOT/'results/patch')
    args=p.parse_args()
    for name in ['learned','random']:
        make_pdf(args.patch/(name+'.png'),args.patch/(name+'.pdf'),name.title())
    print('Saved learned.pdf and random.pdf (four pages each, assembled 40 cm).')


if __name__=='__main__':main()
