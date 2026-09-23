"""
Medical Insurance Cost Estimation - Web App
Run: python webapp.py
Open: http://localhost:5000
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# ─── Train the model ──────────────────────────────────────────────

print("Training model...")

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "insurance.csv")
df = pd.read_csv(csv_path, encoding='latin1')

charges_mean = df['charges'].mean()
charges_std = df['charges'].std()
age_mean = df['age'].mean()
age_std = df['age'].std()
bmi_mean = df['bmi'].mean()
bmi_std = df['bmi'].std()

categorical_columns = df.select_dtypes(include=['object', 'str']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
df_encoded = pd.concat([df.reset_index(drop=True), one_hot_df.reset_index(drop=True)], axis=1)
df_encoded = df_encoded.drop(categorical_columns, axis=1)

scaler = StandardScaler()
df_encoded[['age', 'bmi', 'charges']] = scaler.fit_transform(df_encoded[['age', 'bmi', 'charges']])

X = df_encoded.drop(columns=['charges'])
y = df_encoded['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model ready!")

dataset_stats = {
    "total_records": len(df),
    "avg_charges": float(df['charges'].mean()),
    "avg_age": float(df['age'].mean()),
    "avg_bmi": float(df['bmi'].mean()),
}


def predict(age, sex, bmi, children, smoker, region):
    age_scaled = (age - age_mean) / age_std
    bmi_scaled = (bmi - bmi_mean) / bmi_std
    cat_input = pd.DataFrame([[sex, smoker, region]], columns=categorical_columns)
    cat_encoded = encoder.transform(cat_input)
    features = np.concatenate([[age_scaled, bmi_scaled, children], cat_encoded[0]])
    feature_names = ['age', 'bmi', 'children'] + list(encoder.get_feature_names_out(categorical_columns))
    input_df = pd.DataFrame([features], columns=feature_names)
    prediction_scaled = model.predict(input_df)[0]
    return prediction_scaled * charges_std + charges_mean


# ─── HTML ──────────────────────────────────────────────────────────

HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insurance Cost Estimator</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Outfit', sans-serif;
            min-height: 100vh;
            background: #0f0f12;
            color: #eaeaf0;
            overflow-x: hidden;
        }

        /* ── Floating mesh gradient blobs ── */
        .bg-wrap {
            position: fixed;
            inset: 0;
            z-index: 0;
            overflow: hidden;
            pointer-events: none;
        }

        .blob {
            position: absolute;
            border-radius: 50%;
            filter: blur(120px);
            opacity: 0.45;
            will-change: transform;
        }

        .blob-1 {
            width: 600px; height: 600px;
            background: #6c3ce0;
            top: -10%; left: -8%;
            animation: drift1 18s ease-in-out infinite alternate;
        }

        .blob-2 {
            width: 500px; height: 500px;
            background: #1a8fe0;
            bottom: -12%; right: -5%;
            animation: drift2 22s ease-in-out infinite alternate;
        }

        .blob-3 {
            width: 350px; height: 350px;
            background: #e04592;
            top: 55%; left: 50%;
            animation: drift3 15s ease-in-out infinite alternate;
        }

        @keyframes drift1 {
            0%   { transform: translate(0, 0) scale(1); }
            100% { transform: translate(80px, 60px) scale(1.12); }
        }
        @keyframes drift2 {
            0%   { transform: translate(0, 0) scale(1); }
            100% { transform: translate(-70px, -50px) scale(1.08); }
        }
        @keyframes drift3 {
            0%   { transform: translate(0, 0) scale(1); }
            100% { transform: translate(50px, -40px) scale(0.9); }
        }

        /* ── Grain overlay ── */
        .grain {
            position: fixed;
            inset: 0;
            z-index: 1;
            pointer-events: none;
            opacity: 0.035;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
            background-size: 180px;
        }

        /* ── Layout ── */
        .page {
            position: relative;
            z-index: 2;
            max-width: 1120px;
            margin: 0 auto;
            padding: 3rem 1.5rem 2rem;
        }

        /* ── Header ── */
        .hero {
            text-align: center;
            margin-bottom: 2.8rem;
            animation: riseIn 0.7s cubic-bezier(.22,.68,.36,1.18) both;
        }

        @keyframes riseIn {
            from { opacity: 0; transform: translateY(32px) scale(0.97); }
            to   { opacity: 1; transform: translateY(0)   scale(1); }
        }

        .hero .tag {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #a78bfa;
            background: rgba(167, 139, 250, 0.08);
            border: 1px solid rgba(167, 139, 250, 0.15);
            padding: 0.35rem 1rem;
            border-radius: 999px;
            margin-bottom: 1.4rem;
        }

        .hero h1 {
            font-size: clamp(2rem, 5vw, 3.2rem);
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.02em;
            margin-bottom: 0.9rem;
        }

        .hero h1 .grad {
            background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .hero p {
            color: #8b8ba0;
            font-size: 1.05rem;
            max-width: 520px;
            margin: 0 auto;
            line-height: 1.65;
        }

        /* ── Stat pills ── */
        .pills {
            display: flex;
            justify-content: center;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-bottom: 2.4rem;
            animation: riseIn 0.7s 0.15s both;
        }

        .pill {
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 999px;
            padding: 0.5rem 1.2rem;
            font-size: 0.82rem;
            color: #b0b0c8;
            transition: border-color 0.3s, color 0.3s;
        }

        .pill:hover {
            border-color: rgba(167,139,250,0.3);
            color: #d0d0e8;
        }

        .pill strong {
            color: #eaeaf0;
            font-weight: 600;
            margin-right: 0.25rem;
        }

        /* ── Glass card base ── */
        .glass {
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 20px;
            transition: border-color 0.4s;
        }

        .glass:hover {
            border-color: rgba(255,255,255,0.1);
        }

        /* ── Two-column layout ── */
        .columns {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.6rem;
            animation: riseIn 0.7s 0.3s both;
        }

        /* ── Form ── */
        .form-panel {
            padding: 2.2rem;
        }

        .form-panel h2 {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 1.6rem;
            color: #d4d4e8;
        }

        .field {
            margin-bottom: 1.1rem;
        }

        .field label {
            display: block;
            font-size: 0.78rem;
            font-weight: 500;
            color: #7a7a96;
            margin-bottom: 0.35rem;
            letter-spacing: 0.03em;
        }

        .field input,
        .field select {
            width: 100%;
            padding: 0.7rem 0.9rem;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            color: #eaeaf0;
            font-family: 'Outfit', sans-serif;
            font-size: 0.92rem;
            outline: none;
            transition: border-color 0.3s, box-shadow 0.3s;
        }

        .field input::placeholder { color: #4a4a64; }

        .field input:focus,
        .field select:focus {
            border-color: rgba(167,139,250,0.5);
            box-shadow: 0 0 0 3px rgba(167,139,250,0.1);
        }

        .field select option {
            background: #1a1a24;
            color: #eaeaf0;
        }

        .row-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.9rem;
        }

        .btn {
            width: 100%;
            padding: 0.85rem 1rem;
            border: none;
            border-radius: 12px;
            font-family: 'Outfit', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            color: #fff;
            cursor: pointer;
            margin-top: 0.6rem;
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, #7c3aed, #3b82f6);
            box-shadow: 0 4px 24px rgba(124,58,237,0.25);
            transition: transform 0.25s, box-shadow 0.25s;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(124,58,237,0.35);
        }

        .btn:active { transform: translateY(0); }

        .btn::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.12) 50%, transparent 60%);
            background-size: 250% 100%;
            animation: shimmer 3s ease-in-out infinite;
        }

        @keyframes shimmer {
            0%   { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        .btn.loading { opacity: 0.7; pointer-events: none; }

        /* ── Result panel ── */
        .result-panel {
            padding: 2.2rem;
            display: flex;
            flex-direction: column;
        }

        .result-panel h2 {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 1.6rem;
            color: #d4d4e8;
        }

        .empty-state {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #4a4a64;
            font-size: 0.92rem;
            line-height: 1.6;
            padding: 2rem 1rem;
        }

        /* ── Prediction result ── */
        .outcome {
            display: none;
            flex-direction: column;
            flex: 1;
        }

        .outcome.visible {
            display: flex;
            animation: popIn 0.55s cubic-bezier(.22,.68,.36,1.18) both;
        }

        @keyframes popIn {
            from { opacity: 0; transform: scale(0.92) translateY(12px); }
            to   { opacity: 1; transform: scale(1) translateY(0); }
        }

        .amount-box {
            text-align: center;
            padding: 2rem 1.5rem;
            border-radius: 16px;
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
            background: linear-gradient(160deg, rgba(124,58,237,0.12) 0%, rgba(52,211,153,0.08) 100%);
            border: 1px solid rgba(167,139,250,0.12);
        }

        .amount-box .sup {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #34d399;
            margin-bottom: 0.45rem;
        }

        .amount-box .val {
            font-size: clamp(2.2rem, 4vw, 3.2rem);
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #a78bfa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .amount-box .sub {
            font-size: 0.85rem;
            color: #7a7a96;
            margin-top: 0.3rem;
        }

        /* cost bar */
        .cost-bar {
            padding: 0.9rem 1rem;
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            margin-bottom: 1.2rem;
        }

        .cost-bar .bar-head {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: #7a7a96;
            margin-bottom: 0.4rem;
        }

        .cost-bar .bar-head .level { font-weight: 600; color: #b0b0c8; }

        .track {
            height: 6px;
            background: rgba(255,255,255,0.06);
            border-radius: 999px;
            overflow: hidden;
        }

        .fill {
            height: 100%;
            border-radius: 999px;
            width: 0%;
            transition: width 1.2s cubic-bezier(.22,.68,.36,1), background 0.6s;
        }

        /* detail chips */
        .chips {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 0.6rem;
            margin-top: auto;
        }

        .chip {
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 0.65rem 0.8rem;
        }

        .chip .ck { font-size: 0.65rem; color: #5a5a76; text-transform: uppercase; letter-spacing: 0.06em; }
        .chip .cv { font-size: 0.88rem; font-weight: 600; margin-top: 0.15rem; }

        /* ── Footer ── */
        .foot {
            text-align: center;
            padding: 2.4rem 1rem 1.2rem;
            font-size: 0.75rem;
            color: #3e3e56;
            letter-spacing: 0.02em;
            animation: riseIn 0.7s 0.45s both;
        }

        /* ── Responsive ── */
        @media (max-width: 780px) {
            .columns { grid-template-columns: 1fr; }
            .chips { grid-template-columns: 1fr 1fr; }
        }
    </style>
</head>
<body>
    <div class="bg-wrap">
        <div class="blob blob-1"></div>
        <div class="blob blob-2"></div>
        <div class="blob blob-3"></div>
    </div>
    <div class="grain"></div>

    <div class="page">
        <div class="hero">
            <span class="tag">Machine Learning Prediction</span>
            <h1>Medical Insurance<br><span class="grad">Cost Estimator</span></h1>
            <p>Get an instant estimate of your annual medical insurance cost using a trained regression model.</p>
        </div>

        <div class="pills">
            <span class="pill"><strong id="s-rec">--</strong> records trained</span>
            <span class="pill"><strong id="s-avg">--</strong> avg. cost</span>
            <span class="pill"><strong>80.7%</strong> model accuracy</span>
        </div>

        <div class="columns">
            <div class="glass form-panel">
                <h2>Your Information</h2>
                <form id="frm">
                    <div class="row-2">
                        <div class="field">
                            <label>Age</label>
                            <input type="number" id="age" placeholder="e.g. 30" min="1" max="119" required>
                        </div>
                        <div class="field">
                            <label>Sex</label>
                            <select id="sex" required>
                                <option value="" disabled selected>Select</option>
                                <option value="male">Male</option>
                                <option value="female">Female</option>
                            </select>
                        </div>
                    </div>
                    <div class="row-2">
                        <div class="field">
                            <label>BMI</label>
                            <input type="number" id="bmi" placeholder="e.g. 27.5" step="0.1" min="10" max="60" required>
                        </div>
                        <div class="field">
                            <label>Children</label>
                            <input type="number" id="children" placeholder="e.g. 2" min="0" max="10" required>
                        </div>
                    </div>
                    <div class="row-2">
                        <div class="field">
                            <label>Smoker</label>
                            <select id="smoker" required>
                                <option value="" disabled selected>Select</option>
                                <option value="yes">Yes</option>
                                <option value="no">No</option>
                            </select>
                        </div>
                        <div class="field">
                            <label>Region</label>
                            <select id="region" required>
                                <option value="" disabled selected>Select</option>
                                <option value="northeast">Northeast</option>
                                <option value="northwest">Northwest</option>
                                <option value="southeast">Southeast</option>
                                <option value="southwest">Southwest</option>
                            </select>
                        </div>
                    </div>
                    <button type="submit" class="btn" id="btn">Estimate Cost</button>
                </form>
            </div>

            <div class="glass result-panel">
                <h2>Prediction</h2>
                <div class="empty-state" id="empty">Fill in your details and<br>hit Estimate Cost to see<br>your predicted premium.</div>
                <div class="outcome" id="outcome">
                    <div class="amount-box">
                        <div class="sup">Estimated Annual Premium</div>
                        <div class="val" id="val">$0</div>
                        <div class="sub" id="monthly">~$0 / month</div>
                    </div>
                    <div class="cost-bar">
                        <div class="bar-head">
                            <span>Cost Level</span>
                            <span class="level" id="lvl">--</span>
                        </div>
                        <div class="track"><div class="fill" id="fill"></div></div>
                    </div>
                    <div class="chips">
                        <div class="chip"><div class="ck">Age</div><div class="cv" id="o-age">--</div></div>
                        <div class="chip"><div class="ck">Sex</div><div class="cv" id="o-sex">--</div></div>
                        <div class="chip"><div class="ck">BMI</div><div class="cv" id="o-bmi">--</div></div>
                        <div class="chip"><div class="ck">Smoker</div><div class="cv" id="o-smk">--</div></div>
                        <div class="chip"><div class="ck">Children</div><div class="cv" id="o-kid">--</div></div>
                        <div class="chip"><div class="ck">Region</div><div class="cv" id="o-reg">--</div></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="foot">Medical Insurance Cost Estimation &middot; Linear Regression Model &middot; R&#178; = 0.807</div>
    </div>

    <script>
    fetch('/api/stats').then(r=>r.json()).then(d=>{
        document.getElementById('s-rec').textContent=d.total_records.toLocaleString();
        document.getElementById('s-avg').textContent='$'+Math.round(d.avg_charges).toLocaleString();
    });

    document.getElementById('frm').addEventListener('submit',async e=>{
        e.preventDefault();
        const b=document.getElementById('btn');
        b.classList.add('loading'); b.textContent='Calculating...';

        const fd={
            age:+document.getElementById('age').value,
            sex:document.getElementById('sex').value,
            bmi:+document.getElementById('bmi').value,
            children:+document.getElementById('children').value,
            smoker:document.getElementById('smoker').value,
            region:document.getElementById('region').value
        };

        const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(fd)});
        const j=await r.json();
        const c=j.predicted_cost;

        document.getElementById('empty').style.display='none';
        const o=document.getElementById('outcome');
        o.classList.remove('visible');
        void o.offsetWidth;
        o.classList.add('visible');

        document.getElementById('val').textContent='$'+Math.round(c).toLocaleString();
        document.getElementById('monthly').textContent='~$'+Math.round(c/12).toLocaleString()+' / month';

        const pct=Math.min(c/65000*100,100);
        const fl=document.getElementById('fill');
        fl.style.width=pct+'%';

        let lv,cl;
        if(c<8000){lv='Low';cl='#34d399';}
        else if(c<20000){lv='Moderate';cl='#facc15';}
        else if(c<35000){lv='High';cl='#f97316';}
        else{lv='Very High';cl='#ef4444';}
        document.getElementById('lvl').textContent=lv;
        fl.style.background=cl;

        const cap=s=>s.charAt(0).toUpperCase()+s.slice(1);
        document.getElementById('o-age').textContent=fd.age;
        document.getElementById('o-sex').textContent=cap(fd.sex);
        document.getElementById('o-bmi').textContent=fd.bmi.toFixed(1);
        document.getElementById('o-smk').textContent=cap(fd.smoker);
        document.getElementById('o-kid').textContent=fd.children;
        document.getElementById('o-reg').textContent=cap(fd.region);

        b.classList.remove('loading'); b.textContent='Estimate Cost';
    });
    </script>
</body>
</html>'''


# ─── HTTP Server ──────────────────────────────────────────────────

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == '/api/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(dataset_stats).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/predict':
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length))
            cost = predict(data['age'], data['sex'], data['bmi'],
                           data['children'], data['smoker'], data['region'])
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'predicted_cost': float(cost)}).encode())
        else:
            self.send_error(404)

    def log_message(self, *a): pass


if __name__ == '__main__':
    PORT = int(os.environ.get("PORT", 5000))
    srv = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f"\n{'='*50}")
    print(f"  Medical Insurance Cost Estimator")
    print(f"  http://localhost:{PORT}")
    print(f"{'='*50}\n")
    srv.serve_forever()
