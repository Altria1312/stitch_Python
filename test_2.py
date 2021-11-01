from geopy.distance import geodesic
import piexif

def get_lat_lon(path):
    exif_dict = piexif.load(path)

    lat = exif_dict["GPS"][piexif.GPSIFD.GPSLatitude]
    lon = exif_dict["GPS"][piexif.GPSIFD.GPSLongitude]

    lon = lon[0][0]/lon[0][1] + lon[1][0]/lon[1][1]/60 + lon[2][0]/lon[2][1]/3600
    lat = lat[0][0]/lat[0][1] + lat[1][0]/lat[1][1]/60 + lat[2][0]/lat[2][1]/3600

    return lat, lon

def cale_dist(path1, path2):
    lat1, lon1 = get_lat_lon(path1)
    lat2, lon2 = get_lat_lon(path2)

    dist = geodesic((lat1, lon1), (lat2, lon2)).m

    return dist

def cale_real_roi(height):
    h = 4168
    w = 6252
    focal_length = 25
    dpi = 72

    s = height / focal_length / dpi *25.4 / 100

    H = s * h
    W = s * w

    return H, W

if __name__ == '__main__':
    p1 = r"G:\data\20210817002\20210817002_0005.JPG"
    p2 = r"G:\data\20210817002\20210817002_0006.JPG"

    cale_dist(p1, p2)