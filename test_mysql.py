from sqlalchemy import create_engine, text

# Update with your database credentials
DATABASE_URL = "mysql+pymysql://root:1234@127.0.0.1:3306/altiq_tshirts"

try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM t_shirts LIMIT 5"))
        for row in result:
            print(dict(row._mapping))
    print("✅ MySQL connection successful!")
except Exception as e:
    print("❌ Error connecting to MySQL:", e)
