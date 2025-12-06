"""
FastAPI backend for Phone Comparison Platform
Provides advanced filtering, search, and recommendation endpoints
Neon-ready – no local Postgres required
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor

# ------------------------------------------------------------------
# NEON DATABASE CONFIG – copy / paste ready
# ------------------------------------------------------------------
DB_CONFIG = {
    "host": "ep-twilight-brook-agshqx5x-pooler.c-2.eu-central-1.aws.neon.tech",
    "port": 5432,
    "dbname": "neondb",
    "user": "neondb_owner",
    "password": "npg_iuklm3PF4Itw",
    "sslmode": "require",
}

@contextmanager
def get_db():
    """Database connection context manager – Neon edition"""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

# ------------------------------------------------------------------
# SEARCH HELPER FUNCTIONS
# ------------------------------------------------------------------
def expand_search_query(q: str) -> str:
    """Add common model prefixes for better matching"""
    q_lower = q.lower().strip()

    expansions = {
        'samsung s': 'samsung galaxy s',
        'samsung a': 'samsung galaxy a',
        'samsung z': 'samsung galaxy z',
        'samsung m': 'samsung galaxy m',
        'samsung f': 'samsung galaxy f',
        'apple ': 'apple iphone ',
        'google ': 'google pixel ',
        'huawei p': 'huawei p',
        'huawei mate': 'huawei mate',
        'xiaomi ': 'xiaomi ',
        'redmi ': 'xiaomi redmi ',
        'poco ': 'xiaomi poco ',
        'oneplus ': 'oneplus ',
        'oppo ': 'oppo ',
        'vivo ': 'vivo ',
        'realme ': 'realme ',
        'nothing ': 'nothing phone ',
        'asus rog': 'asus rog phone',
        'sony ': 'sony xperia ',
        'motorola ': 'motorola ',
        'moto ': 'motorola moto ',
        'honor ': 'honor ',
        'zte ': 'zte ',
        'tecno ': 'tecno ',
        'infinix ': 'infinix ',
        'itel ': 'itel ',
    }

    for short, full in expansions.items():
        if q_lower.startswith(short) and not q_lower.startswith(full):
            return full + q_lower[len(short):]
    return q

# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------
app = FastAPI(
    title="Phone Comparison API",
    description="Advanced phone search and filtering API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# RESPONSE MODELS
# ------------------------------------------------------------------
class PhoneBasic(BaseModel):
    id: int
    model_name: str
    brand: str
    price_usd: Optional[float]
    main_image_url: Optional[str]
    screen_size: Optional[float]
    battery_capacity: Optional[int]
    ram_options: Optional[List[int]]
    main_camera_mp: Optional[int]
    chipset: Optional[str]
    antutu_score: Optional[int]
    amazon_link: Optional[str]
    release_year: Optional[int]
    release_month: Optional[int]
    release_day: Optional[int]
    release_date_full: Optional[str]

class PhoneDetail(PhoneBasic):
    price_original: Optional[float]
    currency: Optional[str]
    brand_link: Optional[str]
    weight_g: Optional[float]
    thickness_mm: Optional[float]
    screen_resolution: Optional[str]
    fast_charging_w: Optional[int]
    storage_options: Optional[List[int]]
    video_resolution: Optional[str]
    geekbench_multi: Optional[int]
    gpu_score: Optional[int]
    full_specifications: Optional[dict]
    features: Optional[List[str]]

class SearchResponse(BaseModel):
    total: int
    page: int
    page_size: int
    results: List[PhoneBasic]

class FilterStats(BaseModel):
    total_phones: int
    total_brands: int
    brands: list[dict]
    price_range: dict
    ram_options: list[int]
    battery_range: dict
    release_years: list[int]

# ------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "online", "message": "Phone Comparison API v2.0 – Neon backend", "docs": "/docs"}

@app.get("/phones/search", response_model=SearchResponse)
def search_phones(
    q: Optional[str] = Query(None, description="Search query (model, brand, chipset, specs)"),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    min_ram: Optional[int] = Query(None, ge=2, le=24),
    min_storage: Optional[int] = Query(None, ge=32),
    min_battery: Optional[int] = Query(None, ge=1000),
    min_screen_size: Optional[float] = Query(None, ge=4.0, le=8.0),
    min_camera_mp: Optional[int] = Query(None, ge=8),
    brand: Optional[str] = Query(None),
    min_year: Optional[int] = Query(None, ge=2020, le=2025),
    sort_by: str = Query("release_year"),
    sort_order: str = Query("desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """
    Advanced search with deep JSON spec searching
    Searches in: model, brand, chipset, OS, CPU, GPU, display type, camera features, 
    connectivity (Wi-Fi, Bluetooth, NFC, 5G), sensors, and all nested specifications
    """
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        conditions, params = [], []

        # ENHANCED TEXT SEARCH - searches everywhere including JSON specs
        if q:
            expanded = expand_search_query(q)
            words = expanded.strip().split()
            
            for word in words:
                term = f"%{word}%"
                
                # Build comprehensive search condition for this word
                search_conditions = [
                    # Basic fields
                    "LOWER(model_name) LIKE LOWER(%s)",
                    "LOWER(brand) LIKE LOWER(%s)",
                    "LOWER(chipset) LIKE LOWER(%s)",
                    
                    # JSON Platform specs (OS, CPU, GPU, Chipset details)
                    "LOWER(full_specifications::text) LIKE LOWER(%s)",
                    
                    # Specific JSON paths for better matching
                    "LOWER(full_specifications->'specifications'->'Platform'->>'OS') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Platform'->>'CPU') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Platform'->>'GPU') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Platform'->>'Chipset') LIKE LOWER(%s)",
                    
                    # Display specs (type, refresh rate, protection)
                    "LOWER(full_specifications->'specifications'->'Display'->>'Type') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Display'->>'Size') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Display'->>'Resolution') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Display'->>'Protection') LIKE LOWER(%s)",
                    
                    # Camera specs (main, selfie, features, video)
                    "LOWER(full_specifications->'specifications'->'Main Camera'->>'Single') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Main Camera'->>'Dual') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Main Camera'->>'Triple') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Main Camera'->>'Quad') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Main Camera'->>'Features') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Main Camera'->>'Video') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Selfie Camera'->>'Single') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Selfie Camera'->>'Features') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Selfie Camera'->>'Video') LIKE LOWER(%s)",
                    
                    # Connectivity (Wi-Fi, Bluetooth, USB, NFC, 5G)
                    "LOWER(full_specifications->'specifications'->'Comms'->>'WLAN') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Comms'->>'Bluetooth') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Comms'->>'Positioning') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Comms'->>'NFC') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Comms'->>'Radio') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Comms'->>'USB') LIKE LOWER(%s)",
                    
                    # Network (5G, 4G, bands)
                    "LOWER(full_specifications->'specifications'->'Network'->>'Technology') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Network'->>'2G bands') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Network'->>'3G bands') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Network'->>'4G bands') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Network'->>'5G bands') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Network'->>'Speed') LIKE LOWER(%s)",
                    
                    # Body & Build
                    "LOWER(full_specifications->'specifications'->'Body'->>'Build') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Body'->>'SIM') LIKE LOWER(%s)",
                    
                    # Sound features
                    "LOWER(full_specifications->'specifications'->'Sound'->>'Loudspeaker') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Sound'->>'3.5mm jack') LIKE LOWER(%s)",
                    
                    # Features (sensors, fingerprint, etc)
                    "LOWER(full_specifications->'specifications'->'Features'->>'Sensors') LIKE LOWER(%s)",
                    
                    # Memory
                    "LOWER(full_specifications->'specifications'->'Memory'->>'Card slot') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Memory'->>'Internal') LIKE LOWER(%s)",
                    
                    # Battery
                    "LOWER(full_specifications->'specifications'->'Battery'->>'Type') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Battery'->>'Charging') LIKE LOWER(%s)",
                    
                    # Misc
                    "LOWER(full_specifications->'specifications'->'Misc'->>'Colors') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Misc'->>'Models') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Misc'->>'SAR') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Misc'->>'SAR EU') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Misc'->>'Price') LIKE LOWER(%s)",
                ]
                
                # Wrap all search conditions in OR
                combined_search = "(" + " OR ".join(search_conditions) + ")"
                conditions.append(combined_search)
                
                # Add the search term for each condition (45 times for 45 conditions)
                params.extend([term] * len(search_conditions))

        # Standard filters
        if min_price is not None:
            conditions.append("price_usd >= %s")
            params.append(min_price)
        if max_price is not None:
            conditions.append("price_usd <= %s")
            params.append(max_price)
        if min_ram:
            conditions.append("EXISTS (SELECT 1 FROM unnest(ram_options) AS r WHERE r >= %s)")
            params.append(min_ram)
        if min_storage:
            conditions.append("EXISTS (SELECT 1 FROM unnest(storage_options) AS s WHERE s >= %s)")
            params.append(min_storage)
        if min_battery:
            conditions.append("battery_capacity >= %s")
            params.append(min_battery)
        if min_screen_size:
            conditions.append("screen_size >= %s")
            params.append(min_screen_size)
        if min_camera_mp:
            conditions.append("main_camera_mp >= %s")
            params.append(min_camera_mp)
        if brand:
            conditions.append("LOWER(brand) = LOWER(%s)")
            params.append(brand)
        if min_year:
            conditions.append("release_year >= %s")
            params.append(min_year)

        where = " AND ".join(conditions) if conditions else "1=1"
        
        # Sorting
        valid_sorts = ['price_usd', 'battery_capacity', 'release_year', 'screen_size', 'main_camera_mp', 'antutu_score', 'release_month', 'release_day']
        sort_by = sort_by if sort_by in valid_sorts else 'release_year'
        sort_order = sort_order if sort_order in {'asc', 'desc'} else 'desc'

        # Count total
        count_sql = f"SELECT COUNT(*) as total FROM phones WHERE {where}"
        cursor.execute(count_sql, params)
        total = cursor.fetchone()['total']

        # Get results
        offset = (page - 1) * page_size
        if sort_by == 'release_year':
            order = f"release_year {sort_order} NULLS LAST, release_month {sort_order} NULLS LAST, release_day {sort_order} NULLS LAST"
        else:
            order = f"{sort_by} {sort_order} NULLS LAST"

        sql = f"""
            SELECT id, model_name, brand, price_usd, main_image_url, screen_size, battery_capacity,
                   ram_options, main_camera_mp, chipset, antutu_score, amazon_link,
                   release_year, release_month, release_day, release_date_full
            FROM phones
            WHERE {where}
            ORDER BY {order}
            LIMIT %s OFFSET %s
        """
        cursor.execute(sql, params + [page_size, offset])
        results = cursor.fetchall()

        return {"total": total, "page": page, "page_size": page_size, "results": results}
        
@app.get("/phones/recommend")
def recommend_phones(
    use_case: str = Query(..., description="Use case: gamer, photographer, budget, flagship, battery, performance, balanced"),
    max_price: Optional[float] = Query(None, description="Maximum budget in USD"),
    min_price: Optional[float] = Query(None, description="Minimum price in USD"),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Fast recommendation system with SQL-based sorting
    """
    use_case = use_case.lower()
    
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Base filters
        where = ["price_usd IS NOT NULL"]
        if max_price:
            where.append(f"price_usd <= {max_price}")
        if min_price:
            where.append(f"price_usd >= {min_price}")
        
        # Define ORDER BY and filters for each use case
        order_by = ""
        additional_filters = []
        
        if use_case == "gamer":
            # Gamers want: performance + battery + fast charging
            order_by = """
                COALESCE(antutu_score, 0) * 0.5 + 
                COALESCE(gpu_score, 0) * 0.0015 + 
                COALESCE(battery_capacity, 0) * 0.015 + 
                COALESCE(fast_charging_w, 0) * 0.5 DESC
            """
            additional_filters = [
                "(8 = ANY(ram_options) OR 12 = ANY(ram_options) OR 16 = ANY(ram_options))",
                "battery_capacity >= 4500"
            ]
            
        elif use_case == "photographer":
            # Photographers want: camera MP + battery + screen
            order_by = """
                COALESCE(main_camera_mp, 0) * 2 + 
                COALESCE(battery_capacity, 0) * 0.01 + 
                COALESCE(screen_size, 0) * 10 DESC
            """
            additional_filters = [
                "main_camera_mp >= 48",
                "screen_size >= 6.0"
            ]
            
        elif use_case == "budget":
            # Budget: cheapest with decent specs
            order_by = "price_usd ASC, battery_capacity DESC, main_camera_mp DESC"
            if not max_price:
                where.append("price_usd <= 350")
            additional_filters = ["battery_capacity >= 4000"]
                
        elif use_case == "flagship":
            # Flagship: best performance + camera + charging
            order_by = """
                COALESCE(antutu_score, 0) * 0.4 + 
                COALESCE(main_camera_mp, 0) * 3000 + 
                COALESCE(fast_charging_w, 0) * 1500 DESC
            """
            additional_filters = [
                "(8 = ANY(ram_options) OR 12 = ANY(ram_options) OR 16 = ANY(ram_options))",
                "price_usd >= 500"
            ]
            
        elif use_case == "battery":
            # Battery: capacity + fast charging
            order_by = """
                COALESCE(battery_capacity, 0) * 0.6 + 
                COALESCE(fast_charging_w, 0) * 4 DESC
            """
            additional_filters = ["battery_capacity >= 5000"]
            
        elif use_case == "performance":
            # Performance: AnTuTu + GPU + RAM
            order_by = """
                COALESCE(antutu_score, 0) * 0.5 + 
                COALESCE(gpu_score, 0) * 0.002 DESC
            """
            additional_filters = [
                "(8 = ANY(ram_options) OR 12 = ANY(ram_options) OR 16 = ANY(ram_options))"
            ]
            
        elif use_case == "balanced":
            # Balanced: everything matters
            order_by = """
                COALESCE(antutu_score, 0) * 0.0003 + 
                COALESCE(battery_capacity, 0) * 0.015 + 
                COALESCE(main_camera_mp, 0) * 1.5 + 
                COALESCE(fast_charging_w, 0) * 0.8 DESC
            """
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid use case. Choose: gamer, photographer, budget, flagship, battery, performance, balanced"
            )
        
        where.extend(additional_filters)
        where_clause = " AND ".join(where)
        
        # Single fast query with SQL sorting
        query = f"""
            SELECT 
                id, model_name, brand, price_usd, main_image_url, 
                screen_size, battery_capacity, ram_options, main_camera_mp, 
                chipset, antutu_score, gpu_score, fast_charging_w,
                storage_options, weight_g, release_year, release_date_full
            FROM phones
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT {limit}
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        return {
            "use_case": use_case,
            "max_price": max_price,
            "min_price": min_price,
            "count": len(rows),
            "recommendations": [dict(row) for row in rows]
        }
    
    
@app.get("/phones/latest")
def get_latest_phones(limit: int = Query(20, ge=1, le=50)):
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, model_name, brand, price_usd, main_image_url, screen_size, battery_capacity,
                   ram_options, main_camera_mp, chipset, antutu_score, amazon_link,
                   release_year, release_month, release_day, release_date_full
            FROM phones
            WHERE release_year IS NOT NULL
            ORDER BY release_year DESC NULLS LAST, release_month DESC NULLS LAST, release_day DESC NULLS LAST
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        return {"count": len(rows), "phones": rows}

@app.get("/phones/compare")
def compare_phones(ids: str = Query(..., description="Comma-separated phone IDs (max 4)")):
    phone_ids = [int(x.strip()) for x in ids.split(',') if x.strip()]
    if not phone_ids or len(phone_ids) > 4:
        raise HTTPException(status_code=400, detail="Provide 1-4 phone IDs")
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        placeholders = ','.join(['%s'] * len(phone_ids))
        cursor.execute(f"SELECT * FROM phones WHERE id IN ({placeholders}) ORDER BY price_usd", phone_ids)
        rows = cursor.fetchall()
        if len(rows) != len(phone_ids):
            raise HTTPException(status_code=404, detail="One or more phones not found")
        return {"phones": rows}

@app.get("/phones/{phone_id}", response_model=PhoneDetail)
def get_phone_details(phone_id: int):
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM phones WHERE id = %s", (phone_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Phone not found")
        return row

@app.get("/filters/stats", response_model=FilterStats)
def get_filter_stats():
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # ----------  NEW: global counters  ----------
        cursor.execute("SELECT COUNT(*) AS total_phones FROM phones")
        total_phones = cursor.fetchone()["total_phones"]

        cursor.execute("SELECT COUNT(DISTINCT brand) AS total_brands FROM phones WHERE brand IS NOT NULL")
        total_brands = cursor.fetchone()["total_brands"]
        # --------------------------------------------

        cursor.execute(
            "SELECT brand, COUNT(*) AS count "
            "FROM phones WHERE brand IS NOT NULL "
            "GROUP BY brand ORDER BY count DESC"
        )
        brands = cursor.fetchall()

        cursor.execute(
            "SELECT MIN(price_usd) AS min_price, MAX(price_usd) AS max_price, "
            "ROUND(AVG(price_usd)::numeric, 2) AS avg_price "
            "FROM phones WHERE price_usd IS NOT NULL"
        )
        price_range = cursor.fetchone()

        cursor.execute(
            "SELECT DISTINCT unnest(ram_options) AS ram "
            "FROM phones WHERE ram_options IS NOT NULL ORDER BY ram"
        )
        ram_options = [r["ram"] for r in cursor.fetchall()]

        cursor.execute(
            "SELECT MIN(battery_capacity) AS min_battery, "
            "MAX(battery_capacity) AS max_battery "
            "FROM phones WHERE battery_capacity IS NOT NULL"
        )
        battery_range = cursor.fetchone()

        cursor.execute(
            "SELECT DISTINCT release_year FROM phones "
            "WHERE release_year IS NOT NULL ORDER BY release_year DESC"
        )
        release_years = [r["release_year"] for r in cursor.fetchall()]

        return {
            "total_phones": total_phones,
            "total_brands": total_brands,
            "brands": brands,
            "price_range": price_range,
            "ram_options": ram_options,
            "battery_range": battery_range,
            "release_years": release_years,
        }



@app.get("/brands/{brand}/phones")
def get_brand_phones(
    brand: str,
    sort_by: str = Query("release_year"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
):
    with get_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT COUNT(*) AS total FROM phones WHERE LOWER(brand) = LOWER(%s)", (brand,))
        total = cursor.fetchone()["total"]
        if total == 0:
            raise HTTPException(status_code=404, detail=f"Brand '{brand}' not found")

        offset = (page - 1) * page_size
        valid = ['price_usd', 'release_year', 'battery_capacity', 'model_name']
        sort_field = sort_by if sort_by in valid else 'release_year'

        cursor.execute(f"""
            SELECT id, model_name, brand, price_usd, main_image_url, screen_size, battery_capacity,
                   ram_options, main_camera_mp, chipset, release_year, release_month, release_day, release_date_full
            FROM phones
            WHERE LOWER(brand) = LOWER(%s)
            ORDER BY {sort_field} DESC NULLS LAST
            LIMIT %s OFFSET %s
        """, (brand, page_size, offset))
        rows = cursor.fetchall()

        return {"brand": brand, "total": total, "page": page, "page_size": page_size, "phones": rows}

# ------------------------------------------------------------------
# RUN SERVER
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print('run uvicorn fapi:app --reload')
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

