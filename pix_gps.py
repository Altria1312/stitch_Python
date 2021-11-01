import piexif
import math

def parse_exif(path):
    exif_dict = piexif.load(path)

    lon = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]
    lat = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
    alt = exif_dict["GPS"][piexif.GPSIFD.GPSAltitude]
    foc = exif_dict["Exif"][piexif.ExifIFD.FocalLength]

    lon = lon[0][0]/lon[0][1] + lon[1][0]/lon[1][1]/60 + lon[2][0]/lon[2][1]/3600
    lat = lat[0][0]/lat[0][1] + lat[1][0]/lat[1][1]/60 + lat[2][0]/lat[2][1]/3600
    alt = alt[0]/alt[1]
    foc = foc[0]/foc[1]

    return lon, lat, alt, foc

def gauss_persp_coord(lon, lat):
    # WGS84基准面
    a = 6378137
    b = 6356752.3142
    # 中央经线
    belt = int((lon + 1.5) / 3)
    L0 = belt * 3
    # 子午线弧长
    lat_rad = lat * (math.pi / 180.0)
    e1 = math.sqrt(a**2 - b**2) / a
    m0 = a * (1 - e1*e1)
    m2 = 3 * (e1 * e1 * m0) / 2.0
    m4 = 5 * (e1 * e1 * m2) / 4.0
    m6 = 7 * (e1 * e1 * m4) / 6.0
    m8 = 9 * (e1 * e1 * m6) / 8.0

    a0 = m0 + m2/2.0 + 3*m4/8.0 + 5*m6/16.0 + 35*m8/128.0
    a2 = m2/2.0 + m4/2.0 + 15*m6/32.0 + 7*m8/16.0
    a4 = m4/8.0 + 3*m6/16.0 + 7*m8/32.0
    a6 = m6/32.0 + m8/16.0

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    tan_lat = math.tan(lat_rad)
    eta = (e1 * cos_lat)**2
    X = a0*lat_rad - sin_lat*cos_lat*( (a2-a4+a6) + (2*a4-(16*a6/3))*sin_lat**2 + 16*a6*math.pow(sin_lat, 4)/3.0 )

    l = lon - L0
    N = a / math.sqrt(1 - (e1*sin_lat)**2)

    x = X +\
        0.5 * N * sin_lat * cos_lat * l**2 +\
        N * sin_lat * cos_lat**3 * (5 - tan_lat**2 + 9*eta + 4*eta**2) * l**4 / 24 + \
        N * sin_lat * cos_lat**5 * (61 - 58*tan_lat**4 + 270*eta - 330*eta*tan_lat**2) * l**6 / 720

    y = N * cos_lat * l +\
        N * cos_lat**3 * (1 - tan_lat**2 + eta) * l**3 / 6 + \
        N * cos_lat**5 * (5 - 18*tan_lat**2 + tan_lat**4 + 58*eta*tan_lat**2) * l**5 / 120

    return x, y, L0


if __name__ == '__main__':
    path = r"G:\data\20210817002\20210817002_0005.JPG"
    parse_exif(path)
