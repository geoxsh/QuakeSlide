"""Prepare the small Lombok input bundle from the old project's global data."""
from pathlib import Path
import csv
import shutil
import zipfile
import requests
import geopandas as gpd
from shapely.geometry import box
from osgeo import gdal

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r'E:\Coseismic landslides-20260413\QuakeSlide-20260823\Data')
OUT = ROOT / 'demo' / 'data'
# Small Lombok and neighbouring coastal area; adjust only with corresponding ShakeMap crop.
BBOX = (115.75, -8.95, 117.0, -7.9)

RASTERS = {
    'slope.tif': 'Slope/dtm_slope_merit.dem_m_250m_s0..0cm_2018_v1.0.tif',
    'land_cover.tif': 'land cover-Copernicus LC100/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif',
    'population.tif': 'Population density/gpw-v4-population-density_2020.tif',
    'dem.tif': 'DEM/Median-md75_grd tif/Median-md75_grd.tif',
}

def crop_raster(source, output):
    if not source.exists(): raise FileNotFoundError(source)
    ds = gdal.Warp(str(output), str(source), outputBounds=BBOX,
                   outputBoundsSRS='EPSG:4326', dstSRS='EPSG:4326',
                   creationOptions=['COMPRESS=DEFLATE','TILED=YES'])
    if ds is None: raise RuntimeError(f'Failed to crop {source}')
    ds = None

def crop_vector(source, output):
    if not source.exists(): raise FileNotFoundError(source)
    bounds = gpd.GeoSeries([box(*BBOX)],crs=4326).to_crs(gpd.read_file(source,rows=1).crs).total_bounds
    frame = gpd.read_file(source, bbox=tuple(bounds))
    if frame.crs is None: raise ValueError(f'Missing CRS: {source}')
    frame = frame.to_crs(4326)
    frame = frame[frame.intersects(box(*BBOX))]
    if output.exists(): output.unlink()
    frame.to_file(output, driver='GPKG')
    print(output.name, len(frame), 'features')

def prepare_shakemap():
    local = Path(r'E:\Coseismic landslides-20260413\QuakeSlide-20260517\Results\us1000g3ub\ShakeMap')
    if (local/'PGA.tif').exists() and (local/'PGV.tif').exists():
        crop_raster(local/'PGA.tif', OUT/'PGA.tif')
        crop_raster(local/'PGV.tif', OUT/'PGV.tif')
        print('ShakeMap cropped from',local)
        return
    # Preferred USGS ShakeMap raster.zip for the 5 August event.
    url='https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=us1000g3ub&format=geojson'
    event=requests.get(url,timeout=30);event.raise_for_status()
    products=event.json()['properties']['products']['shakemap']
    product=next(p for p in products if 'download/raster.zip' in p['contents'])
    zip_url=product['contents']['download/raster.zip']['url']
    archive=OUT/'shakemap.zip'
    if not archive.exists():
        with requests.get(zip_url,stream=True,timeout=90) as response:
            response.raise_for_status()
            with archive.open('wb') as f:
                for block in response.iter_content(1<<20): f.write(block)
    with zipfile.ZipFile(archive) as z:
        members={Path(n).name.lower():n for n in z.namelist()}
        for key in ('pga.tif','pgv.tif'):
            if key not in members: raise ValueError(f'{key} missing in USGS raster.zip')
            temp=OUT/f'_source_{key}'
            with z.open(members[key]) as src,temp.open('wb') as dst: shutil.copyfileobj(src,dst)
            crop_raster(temp,OUT/('PGA.tif' if key=='pga.tif' else 'PGV.tif'))
            temp.unlink()
    archive.unlink()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name,relative in RASTERS.items():
        crop_raster(SOURCE/relative,OUT/name)
        print(name,(OUT/name).stat().st_size//1024,'KiB')
    crop_vector(SOURCE/'Global ocean boundary/simplified-water-polygons-split-3857/simplified_water_polygons.shp',OUT/'ocean.gpkg')
    crop_vector(SOURCE/'River network/HydroRIVERS_v10_processed/HydroRIVERS_v10_dischargelargerthan1_riverorderlargerthen2.shp',OUT/'rivers.gpkg')
    lith=gpd.read_file(SOURCE/'Lithology/LiMW_GIS 2015.gdb',layer='GLiM_export',bbox=BBOX)
    lith=lith.to_crs(4326)
    lith.to_parquet(OUT/'lithology.parquet')
    print('lithology.parquet',len(lith),'features')
    cities=SOURCE/'Global cities/worldcities/worldcities.csv'
    with cities.open(encoding='utf-8-sig',newline='') as source,(OUT/'cities.csv').open('w',encoding='utf-8',newline='') as target:
        reader=csv.DictReader(source); writer=csv.DictWriter(target,fieldnames=reader.fieldnames);writer.writeheader()
        lat=next(n for n in reader.fieldnames if n.lower()=='lat')
        lon=next(n for n in reader.fieldnames if n.lower()=='lng')
        for row in reader:
            try:
                if BBOX[0]<=float(row[lon])<=BBOX[2] and BBOX[1]<=float(row[lat])<=BBOX[3]: writer.writerow(row)
            except ValueError: pass
    prepare_shakemap()
    print('Demo inputs prepared:',OUT)

if __name__=='__main__': main()
