"""
SpaceScope: Big Data Analytics of Global Space Missions
Module 1: Data Ingestion
Handles data loading from public sources with automatic fallback
"""

import os
import re
import logging
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONTH_MAP = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}
DATE_PATTERN = re.compile(r'([A-Za-z]{3})\s+(\d{1,2}),\s+(\d{4})')
ISO_PATTERN = re.compile(r'\b(19\d{2}|20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b')
YEAR_PATTERN = re.compile(r'\b(19\d{2}|20\d{2})\b')


def _extract_year_and_date(datum_val):
    """Fast extraction of year (int) and ISO date string from arbitrary datum string"""
    if pd.isna(datum_val):
        return 2020, "2020-01-01"
    s = str(datum_val).strip()
    match = DATE_PATTERN.search(s)
    if match:
        mon, day, yr = match.groups()
        mm = MONTH_MAP.get(mon[:3].capitalize(), '01')
        yr_int = int(yr)
        return yr_int, f"{yr_int}-{mm}-{int(day):02d}"
    
    iso_match = ISO_PATTERN.search(s)
    if iso_match:
        yr, mo, da = iso_match.groups()
        yr_int = int(yr)
        return yr_int, f"{yr_int}-{int(mo):02d}-{int(da):02d}"

    yr_match = YEAR_PATTERN.search(s)
    if yr_match:
        yr_int = int(yr_match.group(1))
        return yr_int, f"{yr_int}-01-01"
    
    return 2020, "2020-01-01"


def _infer_mission_type(detail_val):
    """Categorize mission profile based on payload and mission keywords"""
    d = str(detail_val).lower()
    if any(k in d for k in ['starlink', 'commsat', 'communication', 'telecom', 'satcom', 'iridium', 'intelsat', 'ses-', 'eutelsat', 'express', 'ekspress', 'inmarsat']):
        return 'Communication'
    elif any(k in d for k in ['crew', 'apollo', 'soyuz ms', 'vostok', 'voskhod', 'mercury', 'gemini', 'shenzhou', 'dragon', 'shuttle', 'station', 'iss', 'salyut', 'mir', 'progress']):
        return 'Crewed/Resupply'
    elif any(k in d for k in ['gps', 'glonass', 'beidou', 'galileo', 'navigation', 'navsat']):
        return 'Navigation'
    elif any(k in d for k in ['weather', 'earth', 'observation', 'landsat', 'sentinel', 'gaofen', 'yaogan', 'zy-', 'imaging', 'reconnaissance', 'spy', 'kh-', 'spot']):
        return 'Earth Observation'
    elif any(k in d for k in ['mars', 'lunar', 'moon', 'rover', 'venus', 'deep space', 'hubble', 'telescope', 'probe', 'explorer', 'science', 'scientific', 'test', 'hop', 'prototype']):
        return 'Scientific'
    return 'Commercial'


def standardize_dataframe(raw_df):
    """
    Standardize raw Kaggle Space_Corrected.csv or sample dataset
    into a clean schema with guaranteed columns:
    'id', 'company', 'rocket', 'launch_date', 'year', 'launch_site', 'country',
    'mission_status', 'mission_type', 'detail'
    """
    raw_df = raw_df.reset_index(drop=True)
    clean_df = pd.DataFrame()

    # 0. Deterministic unique integer ID
    if 'id' in raw_df.columns:
        clean_df['id'] = pd.to_numeric(raw_df['id'], errors='coerce').fillna(0).astype(int)
        if (clean_df['id'] <= 0).any() or clean_df['id'].duplicated().any():
            clean_df['id'] = list(range(1, len(raw_df) + 1))
    else:
        clean_df['id'] = list(range(1, len(raw_df) + 1))

    # 1. Company
    if 'Company Name' in raw_df.columns:
        clean_df['company'] = raw_df['Company Name'].fillna('Unknown').astype(str).str.strip()
    elif 'company' in raw_df.columns:
        clean_df['company'] = raw_df['company'].fillna('Unknown').astype(str).str.strip()
    else:
        clean_df['company'] = 'Unknown'

    # 2. Location / Country / Launch site
    if 'Location' in raw_df.columns:
        clean_df['launch_site'] = raw_df['Location'].apply(
            lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown'
        )
        clean_df['country'] = raw_df['Location'].apply(
            lambda x: str(x).split(',')[-1].strip() if pd.notna(x) else 'Unknown'
        )
    else:
        clean_df['launch_site'] = raw_df['launch_site'].fillna('Unknown') if 'launch_site' in raw_df.columns else 'Unknown'
        clean_df['country'] = raw_df['country'].fillna('Unknown') if 'country' in raw_df.columns else 'Unknown'

    # 3. Dates & Year
    if 'Datum' in raw_df.columns:
        parsed_dates = [ _extract_year_and_date(v) for v in raw_df['Datum'] ]
        clean_df['year'] = [p[0] for p in parsed_dates]
        clean_df['launch_date'] = [p[1] for p in parsed_dates]
    elif 'launch_date' in raw_df.columns:
        clean_df['launch_date'] = raw_df['launch_date'].astype(str)
        if 'year' in raw_df.columns:
            clean_df['year'] = pd.to_numeric(raw_df['year'], errors='coerce').fillna(2020).astype(int)
        else:
            clean_df['year'] = clean_df['launch_date'].apply(
                lambda x: int(str(x)[:4]) if len(str(x)) >= 4 and str(x)[:4].isdigit() else 2020
            )
    elif 'year' in raw_df.columns:
        clean_df['year'] = pd.to_numeric(raw_df['year'], errors='coerce').fillna(2020).astype(int)
        clean_df['launch_date'] = clean_df['year'].apply(lambda y: f"{y}-01-01")
    else:
        clean_df['year'] = 2020
        clean_df['launch_date'] = '2020-01-01'

    clean_df['year'] = clean_df['year'].fillna(2020).astype(int)

    # 4. Rocket & Detail
    if 'Detail' in raw_df.columns:
        clean_df['rocket'] = raw_df['Detail'].apply(
            lambda x: str(x).split('|')[0].strip() if pd.notna(x) else 'Unknown'
        )
        clean_df['detail'] = raw_df['Detail'].fillna('').astype(str).str.strip()
    elif 'rocket' in raw_df.columns:
        clean_df['rocket'] = raw_df['rocket'].fillna('Unknown').astype(str).str.strip()
        clean_df['detail'] = raw_df['detail'].fillna(clean_df['rocket']) if 'detail' in raw_df.columns else clean_df['rocket']
    else:
        clean_df['rocket'] = 'Unknown'
        clean_df['detail'] = ''

    # 5. Mission Status
    if 'Status Mission' in raw_df.columns:
        clean_df['mission_status'] = raw_df['Status Mission'].fillna('Unknown').astype(str).str.strip()
    elif 'mission_status' in raw_df.columns:
        clean_df['mission_status'] = raw_df['mission_status'].fillna('Unknown').astype(str).str.strip()
    else:
        clean_df['mission_status'] = 'Success'

    # 6. Mission Type
    if 'mission_type' in raw_df.columns:
        clean_df['mission_type'] = raw_df['mission_type'].fillna('Commercial').astype(str).str.strip()
    elif 'detail' in clean_df.columns:
        clean_df['mission_type'] = clean_df['detail'].apply(_infer_mission_type)
    else:
        clean_df['mission_type'] = 'Commercial'

    return clean_df


class SpaceMissionDataSource:
    """Manages data ingestion from space mission datasets with automatic fallback"""
    
    def __init__(self):
        self.missions_data = None
        self.raw_data_size = 0
        
    def create_sample_dataset(self):
        """Create sample space mission dataset (guaranteed year and schema)"""
        
        sample_data = {
            'company': [
                'USSR', 'USA', 'USSR', 'USA', 'USA', 'USA', 'USSR', 'USA', 'ESA', 'Japan',
                'SpaceX', 'SpaceX', 'SpaceX', 'India', 'SpaceX', 'SpaceX', 'China', 'SpaceX',
                'ISRO', 'SpaceX', 'SpaceX', 'USA', 'China', 'SpaceX', 'SpaceX', 'Rocket Lab',
                'Virgin Orbit', 'Axiom Space', 'SpaceX', 'USA', 'ESA', 'JAXA', 'CNSA',
                'SpaceX', 'Relativity', 'SpaceX', 'China', 'India', 'SpaceX', 'Blue Origin',
                'SpaceX', 'SpaceX', 'China', 'Russia', 'USA', 'SpaceX', 'SpaceX', 'ESA',
                'ISRO', 'SpaceX', 'SpaceX', 'China', 'SpaceX', 'Rocket Lab', 'SpaceX',
                'Virgin Orbit', 'SpaceX', 'USA', 'China', 'SpaceX', 'Relativity Space',
                'SpaceX', 'JAXA', 'SpaceX', 'India', 'SpaceX', 'SpaceX', 'Russia',
                'USA', 'SpaceX', 'ESA', 'SpaceX', 'China', 'SpaceX', 'SpaceX', 'CNSA'
            ],
            'rocket': [
                'R-7', 'Explorer', 'R-7', 'Mercury', 'Gemini', 'Saturn V', 'Proton', 
                'Space Shuttle', 'Ariane', 'H-II', 'Falcon 1', 'Falcon 9', 'Falcon 9',
                'GSLV', 'Falcon 9', 'Falcon 9', 'Long March 2D', 'Falcon Heavy', 'LVM3',
                'Falcon 9', 'Falcon 9', 'Delta IV', 'Long March 5', 'Falcon 9', 'Falcon 9',
                'Electron', 'LauncherOne', 'Axiom', 'Starship', 'Atlas V', 'Ariane 5',
                'H-IIA', 'Long March 3B', 'Falcon 9', 'Terran 1', 'Falcon 9', 'Long March 2C',
                'GSLV Mk III', 'Falcon 9', 'New Shepard', 'Falcon 9', 'Falcon 9', 'Long March 4C',
                'Soyuz', 'Antares', 'Falcon 9', 'Falcon 9', 'Vega', 'Falcon 9 Block 5',
                'Falcon 9', 'Long March 5', 'Electron', 'Falcon 9', 'LauncherOne', 'Falcon 9',
                'Atlas V', 'Long March 2D', 'Falcon 9', 'Chandrayaan', 'Falcon 9', 'Falcon 9',
                'Soyuz 2.1v', 'Vulcan', 'Falcon 9', 'Ariane 6', 'Falcon 9', 'Long March 3A',
                'Falcon 9 Block 5', 'Falcon 9', 'Long March 5'
            ],
            'launch_date': [
                '1957-10-04', '1958-01-31', '1961-04-12', '1962-02-20', '1965-03-23',
                '1969-07-20', '1971-04-19', '1981-04-12', '1984-12-20', '1994-02-04',
                '2006-03-24', '2010-06-04', '2012-05-22', '2013-11-05', '2015-12-22',
                '2016-01-17', '2017-06-15', '2018-02-06', '2019-01-16', '2020-05-30',
                '2020-11-15', '2020-12-03', '2021-03-10', '2021-04-15', '2021-06-10',
                '2021-07-04', '2021-08-04', '2021-09-15', '2021-10-31', '2021-11-20',
                '2021-12-15', '2021-12-18', '2022-01-05', '2022-02-10', '2022-03-02',
                '2022-04-10', '2022-05-14', '2022-06-01', '2022-06-24', '2022-07-01',
                '2022-07-15', '2022-08-01', '2022-08-28', '2022-09-10', '2022-10-15',
                '2022-11-01', '2022-11-15', '2022-12-01', '2022-12-10', '2022-12-20',
                '2021-01-20', '2021-02-15', '2021-03-11', '2021-05-05', '2021-06-30',
                '2021-07-20', '2021-08-26', '2021-09-05', '2021-10-11', '2021-11-13',
                '2021-12-02', '2021-12-14', '2021-12-28', '2022-01-16', '2022-02-20',
                '2022-03-18', '2022-04-22', '2022-05-26', '2022-06-18', '2022-07-22'
            ],
            'launch_site': [
                'Baikonur', 'Cape Canaveral', 'Baikonur', 'Cape Canaveral', 'Cape Canaveral',
                'Kennedy Space Center', 'Baikonur', 'Kennedy Space Center', 'Kourou', 'Tanegashima',
                'Omelek', 'Cape Canaveral', 'Cape Canaveral', 'Satish Dhawan', 'Cape Canaveral',
                'Cape Canaveral', 'Jiuquan', 'Kennedy Space Center', 'Satish Dhawan', 'Kennedy Space Center',
                'Cape Canaveral', 'Cape Canaveral', 'Wenchang', 'Kennedy Space Center', 'Cape Canaveral',
                'Mahia', 'Mojave', 'Houston', 'Starbase', 'Cape Canaveral', 'Kourou', 'Tanegashima',
                'Xichang', 'Kennedy Space Center', 'Long Beach', 'Kennedy Space Center', 'Jiuquan',
                'Satish Dhawan', 'Cape Canaveral', 'New Shepard', 'Kennedy Space Center', 'Cape Canaveral',
                'Taiyuan', 'Baikonur', 'Wallops', 'Kennedy Space Center', 'Cape Canaveral', 'Kourou',
                'Satish Dhawan', 'Kennedy Space Center', 'Cape Canaveral', 'Wenchang', 'Mahia',
                'Kennedy Space Center', 'Mojave', 'Cape Canaveral', 'Cape Canaveral', 'Wenchang',
                'Kennedy Space Center', 'Cape Canaveral', 'Kennedy Space Center', 'Baikonur', 'Canaveral',
                'Kourou', 'Cape Canaveral', 'Wenchang', 'Cape Canaveral', 'Kennedy Space Center'
            ],
            'country': [
                'USSR', 'USA', 'USSR', 'USA', 'USA', 'USA', 'USSR', 'USA', 'France', 'Japan',
                'USA', 'USA', 'USA', 'India', 'USA', 'USA', 'China', 'USA', 'India', 'USA',
                'USA', 'USA', 'China', 'USA', 'USA', 'New Zealand', 'USA', 'USA', 'USA', 'USA',
                'France', 'Japan', 'China', 'USA', 'USA', 'USA', 'China', 'India', 'USA', 'USA',
                'USA', 'USA', 'China', 'Russia', 'USA', 'USA', 'USA', 'France', 'India', 'USA',
                'USA', 'China', 'New Zealand', 'USA', 'USA', 'USA', 'China', 'USA', 'USA', 'USA',
                'Russia', 'USA', 'France', 'USA', 'China', 'USA', 'USA', 'Russia', 'USA', 'USA',
                'France', 'USA', 'China', 'USA', 'USA'
            ],
            'mission_status': [
                'Success', 'Success', 'Success', 'Success', 'Success', 'Success', 'Success',
                'Success', 'Success', 'Success', 'Failed', 'Success', 'Success', 'Success',
                'Success', 'Success', 'Success', 'Success', 'Success', 'Success', 'Success',
                'Success', 'Success', 'Failed', 'Success', 'Success', 'Partial', 'Success',
                'Success', 'Success', 'Success', 'Success', 'Success', 'Success', 'Partial',
                'Success', 'Success', 'Success', 'Success', 'Success', 'Success', 'Success',
                'Success', 'Failed', 'Success', 'Success', 'Success', 'Success', 'Success',
                'Partial', 'Success', 'Success', 'Success', 'Success', 'Success', 'Success',
                'Success', 'Failed', 'Success', 'Success', 'Success', 'Success', 'Success',
                'Success', 'Success', 'Success', 'Success', 'Success', 'Success', 'Success'
            ],
            'mission_type': [
                'Scientific', 'Scientific', 'Crewed', 'Crewed', 'Crewed', 'Crewed', 'Crewed',
                'Communication', 'Communication', 'Earth Observation', 'Communication', 'Communication',
                'Crewed', 'Communication', 'Communication', 'Communication', 'Navigation', 'Scientific',
                'Navigation', 'Crewed', 'Communication', 'Earth Observation', 'Scientific', 'Communication',
                'Communication', 'Communication', 'Communication', 'Scientific', 'Communication', 'Communication',
                'Communication', 'Earth Observation', 'Scientific', 'Scientific', 'Communication',
                'Communication', 'Navigation', 'Navigation', 'Communication', 'Scientific', 'Crewed',
                'Communication', 'Communication', 'Navigation', 'Scientific', 'Communication', 'Communication',
                'Earth Observation', 'Navigation', 'Communication', 'Communication', 'Communication', 'Navigation',
                'Communication', 'Earth Observation', 'Communication', 'Scientific', 'Communication',
                'Navigation', 'Scientific', 'Communication', 'Communication', 'Earth Observation',
                'Communication', 'Scientific', 'Communication', 'Crewed', 'Communication', 'Navigation'
            ]
        }
        # Align all list lengths to avoid ValueError in pandas
        min_len = min(len(v) for v in sample_data.values())
        sample_data = {k: v[:min_len] for k, v in sample_data.items()}
        
        # Explicitly extract and guarantee 'id', 'year' and 'detail' columns
        sample_data['id'] = list(range(1, min_len + 1))
        sample_data['year'] = [int(d.split('-')[0]) for d in sample_data['launch_date']]
        sample_data['detail'] = [f"{r} mission" for r in sample_data['rocket']]

        self.missions_data = pd.DataFrame(sample_data)
        self.raw_data_size = len(self.missions_data)
        
        logger.info(f"✅ Sample dataset created with {self.raw_data_size} records")
        return self.missions_data
    
    def load_from_csv(self, filepath):
        """Load and standardize data from CSV file"""
        try:
            raw_df = pd.read_csv(filepath)
            self.missions_data = standardize_dataframe(raw_df)
            self.raw_data_size = len(self.missions_data)
            logger.info(f"✅ Loaded and standardized {self.raw_data_size} records from {filepath}")
            return self.missions_data
        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            return None
    
    def load_data(self, filepath=None):
        """
        Load data with automatic fallback:
        Attempts to load from CSV (Space_Corrected.csv or given filepath),
        falling back seamlessly to built-in sample dataset if missing or corrupted.
        """
        if filepath:
            candidates = [filepath]
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(base_dir, 'Space_Corrected.csv'),
                'Space_Corrected.csv',
                os.path.join(os.getcwd(), 'Space_Corrected.csv')
            ]

        for path in candidates:
            if os.path.exists(path):
                logger.info(f"🔍 Found dataset file at: {path}")
                df = self.load_from_csv(path)
                if df is not None and not df.empty:
                    return df

        logger.warning("⚠️ No valid CSV file found; falling back to built-in sample dataset")
        return self.create_sample_dataset()

    def get_data_summary(self):
        """Get summary of loaded data"""
        if self.missions_data is None:
            return None
        
        summary = {
            'total_records': len(self.missions_data),
            'columns': list(self.missions_data.columns),
            'date_range': {
                'from': str(self.missions_data['launch_date'].min()),
                'to': str(self.missions_data['launch_date'].max())
            },
            'year_range': {
                'min': int(self.missions_data['year'].min()),
                'max': int(self.missions_data['year'].max())
            },
            'unique_companies': int(self.missions_data['company'].nunique()),
            'unique_rockets': int(self.missions_data['rocket'].nunique()),
            'unique_countries': int(self.missions_data['country'].nunique()),
            'unique_locations': int(self.missions_data['launch_site'].nunique())
        }
        
        return summary
    
    def display_sample_records(self, n=10):
        """Display first n records"""
        if self.missions_data is None:
            logger.warning("⚠️ No data loaded")
            return None
        
        print(f"\n📊 First {n} Space Mission Records:")
        print("=" * 100)
        print(self.missions_data.head(n).to_string(index=False))
        print("=" * 100)
        return self.missions_data.head(n)


def main():
    """Test data ingestion module"""
    print("\n" + "="*80)
    print("🚀 SpaceScope: Data Ingestion Module")
    print("="*80)
    
    data_source = SpaceMissionDataSource()
    data = data_source.load_data()
    summary = data_source.get_data_summary()
    
    print("\n📈 Data Summary:")
    print("-" * 80)
    print(f"Total Records: {summary['total_records']}")
    print(f"Date Range: {summary['date_range']['from']} to {summary['date_range']['to']}")
    print(f"Year Range: {summary['year_range']['min']} to {summary['year_range']['max']}")
    print(f"Unique Companies: {summary['unique_companies']}")
    print(f"Unique Rockets: {summary['unique_rockets']}")
    print(f"Unique Countries: {summary['unique_countries']}")
    print(f"Unique Launch Sites: {summary['unique_locations']}")
    print("-" * 80)
    
    data_source.display_sample_records(5)
    return data_source


if __name__ == "__main__":
    main()
