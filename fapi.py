"""
FastAPI backend for Mobylite Platform
Handles both phone data and user authentication/features
Dual Neon database setup
"""

from fastapi import FastAPI, Query, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from contextlib import contextmanager
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import secrets
import hashlib
import uuid

DB_CONFIG_PHONES = {
    "host": "ep-twilight-brook-agshqx5x-pooler.c-2.eu-central-1.aws.neon.tech",
    "port": 5432,
    "dbname": "neondb",
    "user": "neondb_owner",
    "password": "npg_iuklm3PF4Itw",
    "sslmode": "require",
}

DB_CONFIG_USERS = {
    "host": "ep-shiny-feather-ag2vjll4-pooler.c-2.eu-central-1.aws.neon.tech",
    "port": 5432,
    "dbname": "neondb",
    "user": "neondb_owner",
    "password": "npg_c6nFi5XeBjIY",
    "sslmode": "require",
}

TOKEN_EXPIRATION_HOURS = 24 * 7

@contextmanager
def get_phones_db():
    conn = psycopg2.connect(**DB_CONFIG_PHONES)
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_users_db():
    conn = psycopg2.connect(**DB_CONFIG_USERS)
    try:
        yield conn
    finally:
        conn.close()

def expand_search_query(q: str) -> str:
    q_lower = q.lower().strip()
    expansions = {
        'samsung s': 'samsung galaxy s', 'samsung a': 'samsung galaxy a',
        'samsung z': 'samsung galaxy z', 'samsung m': 'samsung galaxy m',
        'samsung f': 'samsung galaxy f', 'apple ': 'apple iphone ',
        'google ': 'google pixel ', 'huawei p': 'huawei p',
        'huawei mate': 'huawei mate', 'xiaomi ': 'xiaomi ',
        'redmi ': 'xiaomi redmi ', 'poco ': 'xiaomi poco ',
        'oneplus ': 'oneplus ', 'oppo ': 'oppo ', 'vivo ': 'vivo ',
        'realme ': 'realme ', 'nothing ': 'nothing phone ',
        'asus rog': 'asus rog phone', 'sony ': 'sony xperia ',
        'motorola ': 'motorola ', 'moto ': 'motorola moto ',
        'honor ': 'honor ', 'zte ': 'zte ', 'tecno ': 'tecno ',
        'infinix ': 'infinix ', 'itel ': 'itel ',
    }
    for short, full in expansions.items():
        if q_lower.startswith(short) and not q_lower.startswith(full):
            return full + q_lower[len(short):]
    return q

app = FastAPI(
    title="Mobylite API",
    description="Phone comparison and user management API",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

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

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    token: str
    user: dict

class ReviewCreate(BaseModel):
    phone_id: int
    rating: float = Field(ge=0, le=5)
    title: str
    body: str
    pros: Optional[List[str]] = []
    cons: Optional[List[str]] = []

class PriceAlertCreate(BaseModel):
    phone_id: int
    target_price: float

def create_token(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    expiry = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRATION_HOURS)
    
    with get_users_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_sessions (user_id, token, expires_at) VALUES (%s, %s, %s)",
            (user_id, token, expiry)
        )
        conn.commit()
    
    return token

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    
    with get_users_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            "SELECT user_id, expires_at FROM user_sessions WHERE token = %s",
            (token,)
        )
        session = cursor.fetchone()
        
        if not session:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        if datetime.utcnow() > session["expires_at"]:
            cursor.execute("DELETE FROM user_sessions WHERE token = %s", (token,))
            conn.commit()
            raise HTTPException(status_code=401, detail="Token expired")
        
        return str(session["user_id"])

@app.get("/")
def root():
    return {"status": "online", "message": "Mobylite API v3.0 – Dual Neon backend", "docs": "/docs"}

@app.post("/auth/signup", response_model=AuthResponse)
def signup(data: SignupRequest):
    with get_users_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT id FROM users WHERE email = %s", (data.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        cursor.execute(
            "INSERT INTO users (email, password_hash, display_name) VALUES (%s, hash_password(%s), %s) RETURNING id, email, display_name, created_at",
            (data.email, data.password, data.display_name)
        )
        user = cursor.fetchone()
        conn.commit()
        
        token = create_token(str(user["id"]))
        return {"token": token, "user": dict(user)}

@app.post("/auth/login", response_model=AuthResponse)
def login(data: LoginRequest):
    with get_users_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT id, email, display_name, password_hash FROM users WHERE email = %s", (data.email,))
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        cursor.execute("SELECT verify_password(%s, %s) AS valid", (data.password, user["password_hash"]))
        if not cursor.fetchone()["valid"]:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        cursor.execute("UPDATE users SET last_login = NOW() WHERE id = %s", (user["id"],))
        conn.commit()
        
        token = create_token(str(user["id"]))
        del user["password_hash"]
        return {"token": token, "user": user}

@app.get("/auth/me")
def get_current_user(user_id: str = Depends(verify_token)):
    with get_users_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT id, email, display_name, avatar_url, created_at FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

@app.post("/favorites")
def add_favorite(phone_id: int, notes: Optional[str] = None, user_id: str = Depends(verify_token)):
    with get_users_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(
                "INSERT INTO favorites (user_id, phone_id, notes) VALUES (%s, %s, %s) RETURNING id, created_at",
                (user_id, phone_id, notes)
            )
            result = cursor.fetchone()
            conn.commit()
            return {"success": True, "favorite": result}
        except psycopg2.IntegrityError:
            raise HTTPException(status_code=400, detail="Already in favorites")

@app.get("/favorites")
def get_favorites(user_id: str = Depends(verify_token)):
    with get_users_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            "SELECT phone_id, notes, created_at FROM favorites WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
        favorites = cursor.fetchall()
    
    if not favorites:
        return {"favorites": []}
    
    phone_ids = [f["phone_id"] for f in favorites]
    with get_phones_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        placeholders = ','.join(['%s'] * len(phone_ids))
        cursor.execute(f"SELECT * FROM phones WHERE id IN ({placeholders})", phone_ids)
        phones = {p["id"]: dict(p) for p in cursor.fetchall()}
    
    for fav in favorites:
        fav["phone"] = phones.get(fav["phone_id"])
    
    return {"favorites": favorites}

@app.delete("/favorites/{phone_id}")
def remove_favorite(phone_id: int, user_id: str = Depends(verify_token)):
    with get_users_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM favorites WHERE user_id = %s AND phone_id = %s", (user_id, phone_id))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Favorite not found")
    return {"success": True}

@app.post("/reviews")
def create_review(review: ReviewCreate, user_id: str = Depends(verify_token)):
    with get_users_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(
                """INSERT INTO reviews (user_id, phone_id, rating, title, body, pros, cons) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id, created_at""",
                (user_id, review.phone_id, review.rating, review.title, review.body, review.pros, review.cons)
            )
            result = cursor.fetchone()
            conn.commit()
            return {"success": True, "review": result}
        except psycopg2.IntegrityError:
            raise HTTPException(status_code=400, detail="Already reviewed this phone")

@app.get("/reviews/phone/{phone_id}")
def get_phone_reviews(phone_id: int, page: int = 1, page_size: int = 10):
    with get_users_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        offset = (page - 1) * page_size
        cursor.execute(
            """SELECT r.*, u.display_name, u.avatar_url 
               FROM reviews r 
               JOIN users u ON r.user_id = u.id 
               WHERE r.phone_id = %s AND r.is_visible = TRUE 
               ORDER BY r.created_at DESC LIMIT %s OFFSET %s""",
            (phone_id, page_size, offset)
        )
        reviews = cursor.fetchall()
        
        cursor.execute("SELECT COUNT(*) as total FROM reviews WHERE phone_id = %s AND is_visible = TRUE", (phone_id,))
        total = cursor.fetchone()["total"]
        
        cursor.execute("SELECT AVG(rating) as avg_rating FROM reviews WHERE phone_id = %s AND is_visible = TRUE", (phone_id,))
        avg_rating = cursor.fetchone()["avg_rating"] or 0
    
    return {"total": total, "avg_rating": float(avg_rating), "reviews": reviews}

@app.get("/reviews/user")
def get_user_reviews(user_id: str = Depends(verify_token)):
    with get_users_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM reviews WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
        return {"reviews": cursor.fetchall()}

@app.post("/price-alerts")
def create_price_alert(alert: PriceAlertCreate, user_id: str = Depends(verify_token)):
    with get_users_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(
                "INSERT INTO price_alerts (user_id, phone_id, target_price) VALUES (%s, %s, %s) RETURNING id, created_at",
                (user_id, alert.phone_id, alert.target_price)
            )
            result = cursor.fetchone()
            conn.commit()
            return {"success": True, "alert": result}
        except psycopg2.IntegrityError:
            raise HTTPException(status_code=400, detail="Alert already exists")

@app.get("/price-alerts")
def get_price_alerts(user_id: str = Depends(verify_token)):
    with get_users_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM price_alerts WHERE user_id = %s AND is_active = TRUE ORDER BY created_at DESC", (user_id,))
        return {"alerts": cursor.fetchall()}

@app.delete("/price-alerts/{alert_id}")
def delete_price_alert(alert_id: str, user_id: str = Depends(verify_token)):
    with get_users_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM price_alerts WHERE id = %s AND user_id = %s", (alert_id, user_id))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Alert not found")
    return {"success": True}

@app.get("/phones/search", response_model=SearchResponse)
def search_phones(
    q: Optional[str] = Query(None),
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
    with get_phones_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        conditions, params = [], []

        if q:
            expanded = expand_search_query(q)
            words = expanded.strip().split()
            
            for word in words:
                term = f"%{word}%"
                search_conditions = [
                    "LOWER(model_name) LIKE LOWER(%s)", "LOWER(brand) LIKE LOWER(%s)", "LOWER(chipset) LIKE LOWER(%s)",
                    "LOWER(full_specifications::text) LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Platform'->>'OS') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Platform'->>'CPU') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Platform'->>'GPU') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Platform'->>'Chipset') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Display'->>'Type') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Display'->>'Size') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Display'->>'Resolution') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Display'->>'Protection') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Main Camera'->>'Single') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Main Camera'->>'Features') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Comms'->>'WLAN') LIKE LOWER(%s)",
                    "LOWER(full_specifications->'specifications'->'Network'->>'Technology') LIKE LOWER(%s)",
                ]
                conditions.append("(" + " OR ".join(search_conditions) + ")")
                params.extend([term] * len(search_conditions))

        if min_price is not None:
            conditions.append("price_usd >= %s")
            params.append(min_price)
        if max_price is not None:
            conditions.append("price_usd <= %s")
            params.append(max_price)
        if min_ram:
            conditions.append("EXISTS (SELECT 1 FROM unnest(ram_options) AS r WHERE r >= %s)")
            params.append(min_ram)
        if min_battery:
            conditions.append("battery_capacity >= %s")
            params.append(min_battery)
        if brand:
            conditions.append("LOWER(brand) = LOWER(%s)")
            params.append(brand)

        where = " AND ".join(conditions) if conditions else "1=1"
        
        cursor.execute(f"SELECT COUNT(*) as total FROM phones WHERE {where}", params)
        total = cursor.fetchone()['total']

        offset = (page - 1) * page_size
        sql = f"""SELECT id, model_name, brand, price_usd, main_image_url, screen_size, battery_capacity,
                   ram_options, main_camera_mp, chipset, antutu_score, amazon_link,
                   release_year, release_month, release_day, release_date_full
            FROM phones WHERE {where} ORDER BY {sort_by} {sort_order} NULLS LAST LIMIT %s OFFSET %s"""
        cursor.execute(sql, params + [page_size, offset])
        results = cursor.fetchall()

        return {"total": total, "page": page, "page_size": page_size, "results": results}

@app.get("/phones/recommend")
def recommend_phones(use_case: str, max_price: Optional[float] = None, limit: int = 10):
    with get_phones_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        where = ["price_usd IS NOT NULL"]
        if max_price:
            where.append(f"price_usd <= {max_price}")
        
        order_map = {
            "gamer": "COALESCE(antutu_score, 0) * 0.5 + COALESCE(battery_capacity, 0) * 0.015 DESC",
            "photographer": "COALESCE(main_camera_mp, 0) * 2 + COALESCE(battery_capacity, 0) * 0.01 DESC",
            "budget": "price_usd ASC",
            "flagship": "COALESCE(antutu_score, 0) * 0.4 + COALESCE(main_camera_mp, 0) * 3000 DESC",
            "battery": "battery_capacity DESC",
        }
        
        order_by = order_map.get(use_case.lower(), "release_year DESC")
        where_clause = " AND ".join(where)
        
        cursor.execute(f"""SELECT id, model_name, brand, price_usd, main_image_url, screen_size, battery_capacity,
                   ram_options, main_camera_mp, chipset, antutu_score
            FROM phones WHERE {where_clause} ORDER BY {order_by} LIMIT {limit}""")
        return {"use_case": use_case, "recommendations": cursor.fetchall()}

@app.get("/phones/latest")
def get_latest_phones(limit: int = 20):
    with get_phones_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""SELECT id, model_name, brand, price_usd, main_image_url, screen_size, battery_capacity,
                   ram_options, main_camera_mp, chipset, antutu_score, release_year
            FROM phones WHERE release_year IS NOT NULL ORDER BY release_year DESC, release_month DESC LIMIT %s""", (limit,))
        return {"phones": cursor.fetchall()}

@app.get("/phones/{phone_id}", response_model=PhoneDetail)
def get_phone_details(phone_id: int):
    with get_phones_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM phones WHERE id = %s", (phone_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Phone not found")
        return row

@app.get("/filters/stats", response_model=FilterStats)
def get_filter_stats():
    with get_phones_db() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT COUNT(*) AS total_phones FROM phones")
        total_phones = cursor.fetchone()["total_phones"]
        cursor.execute("SELECT COUNT(DISTINCT brand) AS total_brands FROM phones WHERE brand IS NOT NULL")
        total_brands = cursor.fetchone()["total_brands"]
        cursor.execute("SELECT brand, COUNT(*) AS count FROM phones WHERE brand IS NOT NULL GROUP BY brand ORDER BY count DESC")
        brands = cursor.fetchall()
        cursor.execute("SELECT MIN(price_usd) AS min_price, MAX(price_usd) AS max_price FROM phones WHERE price_usd IS NOT NULL")
        price_range = cursor.fetchone()
        cursor.execute("SELECT DISTINCT unnest(ram_options) AS ram FROM phones WHERE ram_options IS NOT NULL ORDER BY ram")
        ram_options = [r["ram"] for r in cursor.fetchall()]
        cursor.execute("SELECT MIN(battery_capacity) AS min_battery, MAX(battery_capacity) AS max_battery FROM phones WHERE battery_capacity IS NOT NULL")
        battery_range = cursor.fetchone()
        cursor.execute("SELECT DISTINCT release_year FROM phones WHERE release_year IS NOT NULL ORDER BY release_year DESC")
        release_years = [r["release_year"] for r in cursor.fetchall()]
        return {
            "total_phones": total_phones, "total_brands": total_brands, "brands": brands,
            "price_range": price_range, "ram_options": ram_options,
            "battery_range": battery_range, "release_years": release_years,
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
