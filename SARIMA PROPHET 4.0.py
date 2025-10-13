"""
Prophet + BOM 하이브리드 모델 v8.0 - Streamlit 앱
두 번째 코드(정상 계산) + JSON 인증 추가
실행: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import io
import base64
import time
import json
from google.oauth2.service_account import Credentials
import gspread
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="원료 예측 시스템 v8.0",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .metric-container {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    h1 {color: #1f77b4;}
    .success-box {background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 0.25rem; padding: 0.75rem; margin: 1rem 0;}
    .cloud-badge {background-color: #007bff; color: white; padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem;}
</style>
""", unsafe_allow_html=True)

# Google Sheets 설정
GOOGLE_SHEETS_CONFIG = {
    'usage': '1lBanCoyOxv71LmXT316mO4XRccMyv5ETKcTcvm8wfvI',
    'inventory': '1k0_QxRBetfP8dFhHH5J478aFPvoMDvn_OPj1428CAzw',
    'bom': '1vdkYQ9tQzuj_juXZPhgEsDdhAXGWqtCejXLZHXNsAws'
}

def get_gspread_client():
    """서비스 계정으로 gspread 클라이언트 생성"""
    try:
        if 'service_account_json' not in st.session_state or not st.session_state.service_account_json:
            return None
        
        service_account_info = st.session_state.service_account_json
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly'
        ]
        
        credentials = Credentials.from_service_account_info(service_account_info, scopes=scopes)
        client = gspread.authorize(credentials)
        return client
    
    except Exception as e:
        st.error(f"❌ 서비스 계정 인증 실패: {str(e)}")
        return None

def read_google_sheet(sheet_id, sheet_name):
    """Google Sheets에서 데이터 읽기 (자동 타입 변환)"""
    try:
        client = get_gspread_client()
        if client is None:
            return None
        
        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        # get_all_values()로 모든 데이터 가져오기
        data = worksheet.get_all_values()
        
        if len(data) == 0:
            return None
        
        if len(data) < 2:
            return None
        
        # 첫 행을 헤더로, 나머지를 데이터로
        headers = data[0]
        rows = data[1:]
        
        # 🔥 중복 빈 컬럼명 처리
        seen = {}
        unique_headers = []
        for i, h in enumerate(headers):
            if h == '' or not h:
                # 빈 컬럼명은 'Unnamed_X'로 변경
                new_name = f'Unnamed_{i}'
                unique_headers.append(new_name)
            elif h in seen:
                # 중복된 컬럼명도 번호 추가
                seen[h] += 1
                unique_headers.append(f'{h}_{seen[h]}')
            else:
                seen[h] = 0
                unique_headers.append(h)
        
        # DataFrame 생성
        df = pd.DataFrame(rows, columns=unique_headers)
        
        # 🔥 자동 타입 변환
        for col in df.columns:
            # 'Unnamed_' 컬럼은 스킵
            if col.startswith('Unnamed_'):
                continue
            
            # 숫자로 변환 시도
            try:
                converted = pd.to_numeric(df[col], errors='coerce')
                # 50% 이상이 숫자면 숫자 컬럼으로 처리
                if converted.notna().sum() > len(df) * 0.5:
                    df[col] = converted.fillna(0)
            except:
                pass
        
        # 'Unnamed_' 컬럼 제거
        df = df.loc[:, ~df.columns.str.startswith('Unnamed_')]
        
        return df
    
    except Exception as e:
        st.error(f"❌ '{sheet_name}' 시트 로드 실패: {str(e)}")
        return None

def read_google_sheet_no_header(sheet_id, sheet_name):
    """헤더 없이 Google Sheets 읽기 (BOM용)"""
    try:
        client = get_gspread_client()
        if client is None:
            return None
        
        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_values()
        
        if len(data) == 0:
            return None
        
        df = pd.DataFrame(data)
        return df
    
    except Exception as e:
        st.error(f"❌ '{sheet_name}' 시트 로드 실패: {str(e)}")
        return None

class BOMHybridModel:
    """BOM 하이브리드 예측 모델 v8.0 (Google Sheets 연동)"""
    
    def __init__(self):
        """모델 초기화"""
        self.hybrid_weights = {
            '대량': {'bom': 0.15, 'prophet': 0.65, 'trend': 0.15, 'ma': 0.05, 
                    'confidence_level': 0.90, 'base_margin': 0.06},
            '중간': {'bom': 0.15, 'prophet': 0.60, 'trend': 0.15, 'ma': 0.10,
                    'confidence_level': 0.85, 'base_margin': 0.10},
            '소량': {'bom': 0.10, 'prophet': 0.60, 'trend': 0.20, 'ma': 0.10,
                    'confidence_level': 0.80, 'base_margin': 0.18}
        }
        
        self.material_corrections = {
            1010101: 1.00, 1030501: 0.95, 1050801: 1.00,
            1010301: 0.73, 1010401: 0.70, 1010201: 0.90,
        }
        
        self.bom_data = {}
        self.bom_available = False
        self.brand_products = {}
    
    def detect_brand(self, product_name):
        """제품명에서 브랜드 자동 감지"""
        if '밥이보약' in product_name:
            return '밥이보약'
        elif '더리얼' in product_name:
            return '더리얼'
        else:
            return '기타'
    
    def load_bom_data_from_sheets(self, sheet_id):
        """Google Sheets에서 BOM 데이터 로드"""
        try:
            with st.spinner("📦 BOM 데이터 로딩 중..."):
                df_raw = read_google_sheet_no_header(sheet_id, '제품 BOM')
                if df_raw is None:
                    self.bom_available = False
                    return False
                
                current_product = None
                
                for idx, row in df_raw.iterrows():
                    first_col = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ''
                    second_col = str(row.iloc[1]).strip() if len(row) > 1 and pd.notna(row.iloc[1]) else ''
                    third_col = str(row.iloc[2]).strip() if len(row) > 2 and pd.notna(row.iloc[2]) else ''
                    
                    # 제품명 행 (첫 번째 열만 값이 있고 나머지 NaN)
                    if first_col and not second_col:
                        current_product = first_col
                        self.bom_data[current_product] = []
                    # 헤더 행 스킵
                    elif first_col.lower() in ['erp 코드', 'erp코드', '원료코드', '품목코드']:
                        continue
                    # 원료 행
                    elif first_col and second_col and third_col and current_product:
                        try:
                            material_code = int(float(first_col))
                            material_name = second_col
                            ratio = float(third_col)
                            
                            self.bom_data[current_product].append({
                                '원료코드': material_code,
                                '원료명': material_name,
                                '배합률': ratio
                            })
                        except (ValueError, TypeError):
                            continue
                
                # 자동 브랜드 매핑
                self.brand_products = {'밥이보약': [], '더리얼': [], '기타': []}
                for product_name in self.bom_data.keys():
                    brand = self.detect_brand(product_name)
                    self.brand_products[brand].append(product_name)
                
                self.bom_available = len(self.bom_data) > 0
                
                if self.bom_available:
                    brand_summary = {brand: len(products) for brand, products in self.brand_products.items()}
                    st.success(
                        f"✅ BOM 데이터 로드 완료!\n"
                        f"- 총 {len(self.bom_data)}개 제품\n"
                        f"- 밥이보약: {brand_summary['밥이보약']}개\n"
                        f"- 더리얼: {brand_summary['더리얼']}개\n"
                        f"- 기타: {brand_summary['기타']}개"
                    )
                    return True
                else:
                    st.warning("⚠️ BOM 데이터가 비어있습니다.")
                    return False
                    
        except Exception as e:
            st.warning(f"⚠️ BOM 데이터 로드 실패: {str(e)}\n기존 방식으로 예측합니다.")
            self.bom_available = False
            return False
    
    def calculate_bom_requirement(self, material_code, production_ton, brand_ratios):
        """BOM 기반 원료 필요량 계산"""
        if not self.bom_available:
            return None
        
        total_requirement = 0.0
        
        for brand, ratio in brand_ratios.items():
            brand_production = production_ton * ratio
            products = self.brand_products.get(brand, [])
            
            if not products:
                continue
            
            material_ratios = []
            for product in products:
                if product in self.bom_data:
                    bom = self.bom_data[product]
                    for item in bom:
                        if item['원료코드'] == material_code:
                            material_ratios.append(item['배합률'])
                            break
            
            if material_ratios:
                avg_ratio = np.mean(material_ratios) / 100
                requirement = brand_production * avg_ratio * 1000
                total_requirement += requirement
        
        return total_requirement if total_requirement > 0 else None
    
    def load_data_from_sheets(self):
        """Google Sheets에서 데이터 로드"""
        try:
            with st.spinner("☁️ Google Sheets에서 데이터 로딩 중..."):
                # 사용량 데이터
                self.df_usage = read_google_sheet(GOOGLE_SHEETS_CONFIG['usage'], '사용량')
                if self.df_usage is None:
                    return False
                
                # 구매량 데이터
                self.df_purchase = read_google_sheet(GOOGLE_SHEETS_CONFIG['usage'], '구매량')
                
                # 생산량 데이터
                self.df_production = read_google_sheet(GOOGLE_SHEETS_CONFIG['usage'], '월별 생산량')
                
                # 브랜드 비중
                self.df_brand = read_google_sheet(GOOGLE_SHEETS_CONFIG['usage'], '브랜드 비중')
                
                # 재고 데이터
                self.df_inventory = read_google_sheet(GOOGLE_SHEETS_CONFIG['inventory'], '재고현황')
                if self.df_inventory is None:
                    return False
                
                # BOM 데이터 (선택적)
                self.load_bom_data_from_sheets(GOOGLE_SHEETS_CONFIG['bom'])
            
            self.prepare_time_series()
            st.success("✅ Google Sheets 데이터 로드 완료!")
            return True
            
        except Exception as e:
            st.error(f"❌ 데이터 로드 실패: {str(e)}")
            return False
    
    def detect_month_columns(self, df):
        """엑셀에서 실제 데이터가 있는 월 컬럼만 감지"""
        month_names = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
        available_months = []
        
        for month in month_names:
            if month in df.columns:
                col_data = df[month]
                valid_data = pd.to_numeric(col_data, errors='coerce').dropna()
                
                if len(valid_data) > 0 and valid_data.sum() > 0:
                    available_months.append(month)
                else:
                    break
        
        return available_months
    
    def prepare_time_series(self):
        """시계열 데이터 준비"""
        self.available_months = self.detect_month_columns(self.df_usage)
        num_months = len(self.available_months)
        
        if num_months == 0:
            st.error("❌ 월 데이터를 찾을 수 없습니다.")
            return
        
        self.months = pd.date_range(start='2025-01-01', periods=num_months, freq='MS')
        self.num_months = num_months
        
        default_prod = [345, 430, 554, 570, 522, 556, 606, 539, 580, 600, 620, 550]
        production_values = []
        
        if len(self.df_production) > 0:
            production_row = self.df_production.iloc[0]
            
            for i, col in enumerate(self.available_months):
                if col in self.df_production.columns:
                    try:
                        val = production_row[col]
                        if isinstance(val, str) and '톤' in val:
                            production_values.append(float(val.replace('톤', '').strip()))
                        elif pd.notna(val) and val != 0:
                            production_values.append(float(val))
                        else:
                            production_values.append(default_prod[i])
                    except:
                        production_values.append(default_prod[i])
                else:
                    production_values.append(default_prod[i])
        
        if len(production_values) == 0:
            production_values = default_prod[:num_months]
        
        while len(production_values) < num_months:
            production_values.append(default_prod[len(production_values)])
        
        production_values = production_values[:num_months]
        
        self.production_ts = pd.DataFrame({'ds': self.months, 'y': production_values})
        
        # 브랜드 비중
        self.brand_ratios = {}
        default_ratios = {'밥이보약': 0.65, '더리얼': 0.33, '기타': 0.02}
        
        for brand in ['밥이보약', '더리얼', '기타']:
            ratios = []
            
            try:
                brand_row = self.df_brand[self.df_brand.iloc[:, 0] == brand]
                
                if not brand_row.empty:
                    for col in self.available_months:
                        if col in self.df_brand.columns:
                            try:
                                val = float(brand_row[col].values[0])
                                ratios.append(val)
                            except:
                                ratios.append(default_ratios[brand])
                        else:
                            ratios.append(default_ratios[brand])
            except:
                pass
            
            if len(ratios) == 0:
                ratios = [default_ratios[brand]] * num_months
            
            while len(ratios) < num_months:
                ratios.append(default_ratios[brand])
            
            self.brand_ratios[brand] = ratios[:num_months]
    
    def safe_float(self, val):
        """안전한 float 변환"""
        try:
            if pd.isna(val) or val is None:
                return 0.0
            return float(val)
        except:
            return 0.0
    
    def classify_material(self, usage_values):
        """원료 분류"""
        avg = np.mean(usage_values) if usage_values else 0
        cv = np.std(usage_values) / avg if avg > 0 else 0
        
        if avg >= 50000 and cv < 0.2:
            return '대량'
        elif avg >= 5000:
            return '중간'
        else:
            return '소량'
    
    def remove_outliers(self, values):
        """IQR 방법으로 이상치 제거"""
        if len(values) < 4:
            return values
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        return [np.median(values) if (v < lower or v > upper) else v for v in values]
    
    def calculate_trend(self, values):
        """트렌드 계산"""
        if len(values) < 2:
            return values[-1] if values else 0
        
        if len(values) >= 3:
            recent = values[-3:]
            trend = recent[-1] + (recent[-1] - recent[0]) / 2
        else:
            trend = values[-1]
        
        weights = np.linspace(0.5, 1.5, len(values))
        weights = weights / weights.sum()
        weighted = np.average(values, weights=weights)
        
        return trend * 0.7 + weighted * 0.3
    
    def train_prophet_simple(self, data, material_type):
        """단순화된 Prophet"""
        try:
            if len(data) < 2 or data['y'].sum() == 0:
                return None
            
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.1 if material_type == '대량' else 0.15,
                interval_width=self.hybrid_weights[material_type]['confidence_level'],
                uncertainty_samples=50
            )
            
            if 'production' in data.columns and material_type != '소량':
                model.add_regressor('production', standardize=True)
            
            model.fit(data)
            return model
        except:
            return None
    
    def predict_material(self, material_code, material_name, usage_values, 
                        next_month_production, brand_ratios):
        """개별 원료 예측 (하이브리드 + 안전장치)"""
        try:
            if sum(usage_values) == 0:
                return 0, (0, 0), 'N/A'
            
            cleaned = self.remove_outliers(usage_values)
            material_type = self.classify_material(cleaned)
            weights = self.hybrid_weights[material_type]
            
            avg_prod = np.mean(self.production_ts['y'].values)
            prod_ratio = next_month_production / avg_prod if avg_prod > 0 else 1
            
            historical_max = max(cleaned) if cleaned else 0
            
            # 1. BOM 기반 예측
            bom_pred = self.calculate_bom_requirement(material_code, next_month_production, brand_ratios)
            
            # 안전장치: BOM이 과거 최대값의 2배 초과하면 무시
            bom_safe = False
            if bom_pred is not None and bom_pred > 0:
                if historical_max > 0 and bom_pred > historical_max * 2:
                    bom_pred = None
                else:
                    bom_safe = True
            
            # 2. Prophet 예측
            prophet_pred = np.mean(cleaned[-3:]) * prod_ratio
            try:
                train_data = pd.DataFrame({
                    'ds': self.months[:len(cleaned)],
                    'y': cleaned,
                    'production': self.production_ts['y'].values[:len(cleaned)]
                })
                
                prophet_model = self.train_prophet_simple(train_data, material_type)
                
                if prophet_model:
                    next_month_date = self.months[len(cleaned) - 1] + pd.DateOffset(months=1)
                    future = pd.DataFrame({
                        'ds': [next_month_date],
                        'production': [next_month_production]
                    })
                    forecast = prophet_model.predict(future)
                    prophet_pred = max(0, forecast['yhat'].values[0])
            except:
                pass
            
            # 3. 트렌드 예측
            trend_pred = self.calculate_trend(cleaned) * prod_ratio
            
            # 4. 이동평균
            ma_pred = np.mean(cleaned[-3:]) * prod_ratio
            
            # 5. 하이브리드 앙상블
            if bom_pred is not None and bom_pred > 0 and bom_safe:
                final_pred = (
                    bom_pred * weights['bom'] +
                    prophet_pred * weights['prophet'] +
                    trend_pred * weights['trend'] +
                    ma_pred * weights['ma']
                )
                confidence = 'BOM+AI'
            else:
                total_weight = weights['prophet'] + weights['trend'] + weights['ma']
                final_pred = (
                    prophet_pred * (weights['prophet'] / total_weight) +
                    trend_pred * (weights['trend'] / total_weight) +
                    ma_pred * (weights['ma'] / total_weight)
                )
                confidence = 'AI only' if bom_pred is None else 'AI (BOM차단)'
            
            # 6. 보정
            if material_code in self.material_corrections:
                final_pred *= self.material_corrections[material_code]
            
            # 브랜드 보정
            if '닭' in str(material_name) or 'MDCM' in str(material_name):
                final_pred *= (1 + (brand_ratios['밥이보약'] - 0.62) * 0.2)
            elif '소고기' in str(material_name) or '연어' in str(material_name):
                final_pred *= (1 + (brand_ratios['더리얼'] - 0.35) * 0.3)
            
            # 7. 신뢰구간
            margin = weights['base_margin']
            lower = final_pred * (1 - margin)
            upper = final_pred * (1 + margin)
            
            return final_pred, (lower, upper), confidence
            
        except:
            return np.mean(usage_values[-3:]) if usage_values else 0, (0, 0), 'N/A'
    
    def get_inventory(self, material_code):
        """재고 조회"""
        try:
            row = self.df_inventory[self.df_inventory['품목코드'] == material_code]
            if not row.empty:
                for col in reversed(row.columns):
                    val = row.iloc[0][col]
                    if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
                        return float(val)
        except:
            pass
        return 0
    
    def predict_all(self, next_month_production, brand_ratios):
        """전체 예측"""
        results = []
        total = len(self.df_usage)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
        
        start_time = time.time()
        
        for idx, row in self.df_usage.iterrows():
            progress = (idx + 1) / total
            progress_bar.progress(progress)
            status_text.text(f'예측 중... {idx + 1}/{total} ({progress*100:.1f}%)')
            
            if idx > 0:
                elapsed = time.time() - start_time
                eta = elapsed / (idx + 1) * (total - idx - 1)
                time_text.text(f'예상 남은 시간: {eta:.0f}초')
            
            material_code = row['원료코드']
            material_name = row['품목명']
            
            usage_values = []
            for col in self.available_months:
                if col in row.index:
                    usage_values.append(self.safe_float(row[col]))
            
            usage_pred, (lower, upper), confidence = self.predict_material(
                material_code, material_name, usage_values,
                next_month_production, brand_ratios
            )
            
            inventory = self.get_inventory(material_code)
            safety_stock = usage_pred * 0.15
            purchase = max(0, usage_pred - inventory + safety_stock)
            
            category = self.classify_material(usage_values)
            
            range_width = ((upper - lower) / usage_pred * 100) if usage_pred > 0 else 0
            
            results.append({
                '원료코드': material_code,
                '품목명': material_name,
                '예측_사용량': round(usage_pred, 2),
                '사용량_하한': round(lower, 2),
                '사용량_상한': round(upper, 2),
                '신뢰구간_폭': f"±{range_width/2:.1f}%",
                '예측_구매량': round(purchase, 2),
                '현재_재고': round(inventory, 2),
                '원료_분류': category,
                '예측_방식': confidence
            })
        
        progress_bar.empty()
        status_text.empty()
        time_text.empty()
        
        total_time = time.time() - start_time
        st.success(f"✅ 예측 완료! (소요시간: {total_time:.1f}초)")
        
        return pd.DataFrame(results)

def create_charts(df):
    """차트 생성"""
    fig_pie = px.pie(
        df['원료_분류'].value_counts().reset_index(),
        values='count', names='원료_분류', title="원료 분류별 분포",
        color_discrete_map={'대량': '#1f77b4', '중간': '#ff7f0e', '소량': '#2ca02c'}
    )
    
    top10 = df.nlargest(10, '예측_사용량')
    fig_bar = px.bar(
        top10, x='예측_사용량', y='품목명', orientation='h',
        title="TOP 10 예측 사용량", color='원료_분류', text='예측_사용량'
    )
    fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    if '예측_방식' in df.columns:
        fig_method = px.pie(
            df['예측_방식'].value_counts().reset_index(),
            values='count', names='예측_방식', title="예측 방식 분포",
            color_discrete_map={'BOM+AI': '#28a745', 'AI only': '#ffc107', 'AI (BOM차단)': '#dc3545'}
        )
    else:
        fig_method = None
    
    return fig_pie, fig_bar, fig_method

def get_download_link(df):
    """다운로드 링크 생성"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='예측결과', index=False)
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="예측결과_v8.0.xlsx">📥 엑셀 다운로드</a>'

def main():
    """메인 애플리케이션"""
    
    # 헤더
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("☁️ 원료 예측 시스템 v8.0")
        st.markdown("**Google Sheets 실시간 연동** (Prophet 65% + BOM 15% + 안전장치)")
    with col2:
        st.markdown("""
        <div class="success-box">
        <b>v8.0 신기능</b><br>
        • <span class="cloud-badge">Google Sheets</span><br>
        • JSON 키 업로드<br>
        • 실시간 데이터<br>
        • 자동 새로고침
        </div>
        """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 서비스 계정 인증
        has_auth = 'service_account_json' in st.session_state and st.session_state.service_account_json
        
        if not has_auth:
            st.subheader("🔐 Google 인증")
            st.info("**서비스 계정 JSON 키 파일을 업로드하세요**")
            
            json_file = st.file_uploader("JSON 키 파일 선택", type=['json'])
            
            if json_file is not None:
                try:
                    json_content = json_file.read()
                    service_account_json = json.loads(json_content)
                    
                    required_fields = ['type', 'project_id', 'private_key', 'client_email']
                    if all(field in service_account_json for field in required_fields):
                        st.session_state.service_account_json = service_account_json
                        st.success(f"✅ 인증 완료!")
                        st.rerun()
                    else:
                        st.error("❌ 유효하지 않은 서비스 계정 파일입니다.")
                except Exception as e:
                    st.error(f"❌ 오류: {str(e)}")
            
            with st.expander("📖 도움말"):
                st.markdown("""
                **Google Sheets 권한 설정:**
                
                1. Google Sheets 파일 열기
                2. "공유" 클릭
                3. 서비스 계정 이메일 추가
                4. 권한: "뷰어"
                5. 3개 파일 모두 적용
                """)
            return
        
        else:
            email = st.session_state.service_account_json.get('client_email', 'Unknown')
            st.success("🔐 인증 완료")
            st.caption(f"📧 {email}")
            
            if st.button("🗑️ 인증 해제"):
                del st.session_state.service_account_json
                if 'model' in st.session_state:
                    del st.session_state.model
                st.rerun()
        
        st.markdown("---")
        
        st.subheader("☁️ 데이터 소스")
        st.info("**Google Sheets 연동됨!**")
        
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            if 'model' in st.session_state:
                del st.session_state.model
            st.rerun()
        
        st.markdown("---")
        
        st.subheader("📝 예측 조건")
        
        production = st.number_input("생산 계획 (톤)", min_value=100.0, max_value=1000.0, value=600.0, step=10.0)
        
        st.markdown("**브랜드 비중 (%)**")
        col1, col2 = st.columns(2)
        with col1:
            bob = st.slider("밥이보약", 0, 100, 60, 1)
        with col2:
            real = st.slider("더리얼", 0, 100, 35, 1)
        
        etc = 100 - bob - real
        if etc < 0:
            st.error("비중 합이 100%를 초과!")
            return
        
        st.metric("기타", f"{etc}%")
        
        brand_ratios = {'밥이보약': bob/100, '더리얼': real/100, '기타': etc/100}
        
        st.markdown("---")
        
        predict_btn = st.button("🔮 예측 실행", type="primary", use_container_width=True)
        
        with st.expander("📊 모델 정보"):
            st.markdown("""
            **v8.0 하이브리드 구성**
            
            - Prophet: 60-65% ⭐
            - BOM: 10-15%
            - 트렌드: 15-20%
            - 이동평균: 5-10%
            
            **안전장치**
            - BOM 과대예측 자동 차단
            """)
    
    # 메인 영역
    if 'model' not in st.session_state:
        st.session_state.model = BOMHybridModel()
        st.session_state.data_loaded = False
    
    model = st.session_state.model
    
    if not st.session_state.data_loaded:
        if model.load_data_from_sheets():
            st.session_state.data_loaded = True
        else:
            return
    
    if st.session_state.data_loaded:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("원료 수", f"{len(model.df_usage):,}")
        with col2:
            st.metric("데이터 기간", f"1-{model.num_months}월")
        with col3:
            st.metric("생산 계획", f"{production:.0f}톤")
        with col4:
            if model.bom_available:
                st.metric("BOM 제품", f"{len(model.bom_data)}개", delta="통합됨")
            else:
                st.metric("BOM 상태", "미사용")
        
        if predict_btn:
            st.markdown("---")
            st.header("📈 예측 결과")
            
            predictions = model.predict_all(production, brand_ratios)
            
            if predictions is not None and not predictions.empty:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("총 예측 사용량", f"{predictions['예측_사용량'].sum():,.0f}")
                with col2:
                    st.metric("총 예측 구매량", f"{predictions['예측_구매량'].sum():,.0f}")
                with col3:
                    avg_range = predictions['신뢰구간_폭'].apply(lambda x: float(x.replace('±', '').replace('%', ''))).mean()
                    st.metric("평균 신뢰구간", f"±{avg_range:.1f}%")
                with col4:
                    if model.bom_available:
                        bom_count = len(predictions[predictions['예측_방식']=='BOM+AI'])
                        st.metric("BOM 적용", f"{bom_count}개")
                    else:
                        st.metric("예측 방식", "AI only")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 차트", "📋 데이터", "🎯 TOP 20", "📥 다운로드"])
                
                with tab1:
                    fig_pie, fig_bar, fig_method = create_charts(predictions)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col2:
                        if fig_method:
                            st.plotly_chart(fig_method, use_container_width=True)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with tab2:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        categories = st.multiselect("분류 필터", ['대량', '중간', '소량'], ['대량', '중간', '소량'])
                    with col2:
                        if model.bom_available:
                            methods = st.multiselect("예측 방식", ['BOM+AI', 'AI only', 'AI (BOM차단)'], ['BOM+AI', 'AI only', 'AI (BOM차단)'])
                        else:
                            methods = ['AI only']
                    with col3:
                        search = st.text_input("원료명 검색")
                    
                    filtered = predictions[predictions['원료_분류'].isin(categories)]
                    if model.bom_available:
                        filtered = filtered[filtered['예측_방식'].isin(methods)]
                    if search:
                        filtered = filtered[filtered['품목명'].str.contains(search, case=False, na=False)]
                    
                    st.dataframe(filtered, use_container_width=True, height=400)
                    st.caption(f"총 {len(filtered)}개 원료")
                
                with tab3:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🔝 사용량 TOP 20")
                        display_cols = ['품목명', '예측_사용량', '신뢰구간_폭', '원료_분류']
                        if model.bom_available:
                            display_cols.append('예측_방식')
                        top20_usage = predictions.nlargest(20, '예측_사용량')[display_cols]
                        st.dataframe(top20_usage, use_container_width=True)
                    with col2:
                        st.subheader("🛒 구매량 TOP 20")
                        display_cols = ['품목명', '예측_구매량', '현재_재고', '원료_분류']
                        if model.bom_available:
                            display_cols.append('예측_방식')
                        top20_purchase = predictions.nlargest(20, '예측_구매량')[display_cols]
                        st.dataframe(top20_purchase, use_container_width=True)
                
                with tab4:
                    st.subheader("📥 결과 다운로드")
                    st.markdown(get_download_link(predictions), unsafe_allow_html=True)
                    
                    csv = predictions.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📄 CSV 다운로드", csv, "predictions_v8.0.csv", "text/csv")
                    
                    bom_status = f"BOM 통합 ({len(model.bom_data)}개 제품)" if model.bom_available else "BOM 미사용"
                    st.info(f"""
                    **파일 정보**
                    - 원료: {len(predictions)}개
                    - 모델: v8.0 (Google Sheets)
                    - BOM: {bom_status}
                    - 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    """)

if __name__ == "__main__":
    main()
