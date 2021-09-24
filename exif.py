import piexif
from PIL import Image
from PIL.ExifTags import TAGS


if __name__ == "__main__":
    img = Image.open("./img.jpg")
    # for tag, value in img._getexif().items():
    #     print(TAGS.get(tag, tag), value)
    w, h = img.size
    piexif.helper
    exif_dict = piexif.load("./img.jpg")
    exif_dict["0th"][piexif.ImageIFD.Make] = "Me"
    exif_dict["0th"][piexif.ImageIFD.XResolution] = (w, 1)
    exif_dict["0th"][piexif.ImageIFD.YResolution] = (h, 1)
    exif_dict["0th"][piexif.ImageIFD.Software] = "mypiexif"

    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = u"2019:09:29 10:10:10"
    exif_dict["Exif"][piexif.ExifIFD.LensMake] = "LLL"
    exif_dict["Exif"][piexif.ExifIFD.Sharpness] = 65535
    exif_dict["Exif"][piexif.ExifIFD.LensSpecification] = ((1, 1),
                                                           (1, 1),
                                                           (1, 1),
                                                           (1, 1))

    exif_dict["GPS"][piexif.GPSIFD.GPSVersionID] = (2, 0, 0, 0)
    exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = 1
    exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = (23456, 100)
    exif_dict["GPS"][piexif.GPSIFD.GPSDateStamp] = u"2099:09:29 10:10:10"
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = "N"
    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = ((24, 1), (54, 1), (22334, 1000))
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = "E"
    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = ((24, 2), (54, 1), (223340, 10000))



    exif = piexif.dump(exif_dict)
    img.save("./exif.jpeg", exif=exif)

    pass


