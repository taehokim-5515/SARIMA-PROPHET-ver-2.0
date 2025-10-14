"""
Prophet + BOM 하이브리드 모델 v7.1 - Google Sheets 연동
실제 패턴(Prophet 65%) 중심, BOM 참고용(15%)
안전장치로 BOM 과대예측 방지
정확도 대폭 향상

실행 전 설치:
pip install streamlit pandas numpy prophet plotly gspread google-auth openpyxl

실행: streamlit run app_google_sheets.py
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
import gspread
from google.oauth2.service_account import Credentials
import json

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="원료 사용량 예측 시나리오 v7.1 (Google Sheets)",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    h1 {
        color: #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .bom-badge {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    .google-sheets-badge {
        background-color: #34a853;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 스프레드시트 URL (하드코딩)
SPREADSHEET_URLS = {
    'bom': '1vdkYQ9tQzuj_juXZPhgEsDdhAXGWqtCejXLZHXNsAws',
    'usage': '1lBanCoyOxv71LmXT316mO4XRccMyv5ETKcTcvm8wfvI',
    'inventory': '1k0_QxRBetfP8dFhHH5J478aFPvoMDvn_OPj1428CAzw'
}

class GoogleSheetsConnector:
    """Google Sheets 연결 관리자 (캐싱 추가)"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self._cache = {}  # 캐시 저장소
    
    def connect(self, credentials_json):
        """Google Sheets API 연결"""
        try:
            # JSON 파싱
            if isinstance(credentials_json, bytes):
                credentials_dict = json.loads(credentials_json.decode('utf-8'))
            else:
                credentials_dict = json.loads(credentials_json)
            
            # 인증 범위 설정
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets.readonly',
                'https://www.googleapis.com/auth/drive.readonly'
            ]
            
            # 자격증명 생성
            creds = Credentials.from_service_account_info(
                credentials_dict,
                scopes=scopes
            )
            
            # gspread 클라이언트 생성
            self.client = gspread.authorize(creds)
            self.connected = True
            
            return True
            
        except Exception as e:
            st.error(f"❌ Google Sheets 연결 실패: {str(e)}")
            self.connected = False
            return False
    
    @st.cache_data(ttl=300)  # 5분간 캐시 유지
    def read_sheet(_self, spreadsheet_id, sheet_name):
        """스프레드시트에서 데이터 읽기 (캐싱 버전)"""
        try:
            # 캐시 키 생성
            cache_key = f"{spreadsheet_id}_{sheet_name}"
            
            # 캐시에 있으면 반환
            if cache_key in _self._cache:
                st.info(f"💾 캐시에서 '{sheet_name}' 시트 로드 (API 호출 없음)")
                return _self._cache[cache_key].copy()
            
            if not _self.connected:
                raise Exception("Google Sheets에 연결되지 않았습니다.")
            
            st.info(f"☁️ '{sheet_name}' 시트 읽는 중... (API 호출)")
            
            # 스프레드시트 열기
            spreadsheet = _self.client.open_by_key(spreadsheet_id)
            
            # 워크시트 선택
            worksheet = spreadsheet.worksheet(sheet_name)
            
            # 데이터를 DataFrame으로 변환
            data = worksheet.get_all_values()
            
            if not data:
                return pd.DataFrame()
            
            # 첫 행을 헤더로 사용
            df = pd.DataFrame(data[1:], columns=data[0])
            
            # 중복 컬럼명 처리
            if len(df.columns) != len(set(df.columns)):
                new_columns = []
                col_count = {}
                
                for col in df.columns:
                    if col == '' or col is None:
                        col = 'Unnamed'
                    
                    if col in col_count:
                        col_count[col] += 1
                        new_columns.append(f"{col}_{col_count[col]}")
                    else:
                        col_count[col] = 0
                        new_columns.append(col)
                
                df.columns = new_columns
            
            # 캐시에 저장
            _self._cache[cache_key] = df.copy()
            
            return df
            
        except Exception as e:
            st.error(f"❌ 시트 '{sheet_name}' 읽기 실패: {str(e)}")
            return None
    
    def clear_cache(self):
        """캐시 초기화"""
        self._cache = {}
        st.success("✅ 캐시가 초기화되었습니다!")
    
    def read_sheet_with_header(self, spreadsheet_id, sheet_name, header_row=0):
        """헤더 위치를 지정하여 스프레드시트 읽기"""
        try:
            if not self.connected:
                raise Exception("Google Sheets에 연결되지 않았습니다.")
            
            spreadsheet = self.client.open_by_key(spreadsheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)
            
            # 모든 데이터 가져오기
            data = worksheet.get_all_values()
            
            if len(data) <= header_row:
                return pd.DataFrame()
            
            # 지정된 행을 헤더로 사용
            df = pd.DataFrame(data[header_row + 1:], columns=data[header_row])
            
            return df
            
        except Exception as e:
            st.error(f"❌ 시트 읽기 실패: {str(e)}")
            return None

class BOMHybridModel:
    """BOM 하이브리드 예측 모델 v7.1 (Google Sheets 버전)"""
    
    def __init__(self, sheets_connector):
        """모델 초기화"""
        self.sheets = sheets_connector
        
        # 하이브리드 가중치 (실제 패턴 중심, BOM 참고용)
        self.hybrid_weights = {
            '대량': {
                'bom': 0.15,       # BOM 기반 (참고용) - 축소
                'prophet': 0.65,   # Prophet (실제 패턴) - 대폭 강화!
                'trend': 0.15,     # 트렌드
                'ma': 0.05,        # 이동평균
                'confidence_level': 0.90,
                'base_margin': 0.06
            },
            '중간': {
                'bom': 0.15,
                'prophet': 0.60,
                'trend': 0.15,
                'ma': 0.10,
                'confidence_level': 0.85,
                'base_margin': 0.10
            },
            '소량': {
                'bom': 0.10,
                'prophet': 0.60,
                'trend': 0.20,
                'ma': 0.10,
                'confidence_level': 0.80,
                'base_margin': 0.18
            }
        }
        
        # 검증된 보정계수
        self.material_corrections = {
            1010101: 1.00,   # 닭고기 MDCM
            1030501: 0.95,   # 콘그릿츠
            1050801: 1.00,   # 녹색 완두
            1010301: 0.73,   # 소고기 분쇄육
            1010401: 0.70,   # 연어
            1010201: 0.90,   # 오리고기
        }
        
        # BOM 데이터
        self.bom_data = {}
        self.bom_available = False
        self.brand_products = {}
    
    def detect_brand(self, product_name):
        """제품명에서 브랜드 자동 감지"""
        product_name_lower = str(product_name).lower()
        
        if '밥이보약' in product_name:
            return '밥이보약'
        elif '더리얼' in product_name:
            return '더리얼'
        elif '마푸' in product_name or '프라임펫' in product_name or \
             '닥터썸업' in product_name or '펫후' in product_name or \
             '용가리' in product_name or '맥시칸' in product_name:
            return '기타'
        else:
            return '기타'
    
    def load_bom_data(self):
        """Google Sheets에서 BOM 데이터 로드"""
        try:
            with st.spinner("📦 BOM 데이터 로딩 중 (Google Sheets)..."):
                # BOM 시트 읽기 (헤더 없음)
                spreadsheet = self.sheets.client.open_by_key(SPREADSHEET_URLS['bom'])
                worksheet = spreadsheet.worksheet('제품 BOM')
                data = worksheet.get_all_values()
                
                # DataFrame 생성 (헤더 없음)
                df_raw = pd.DataFrame(data)
                
                # BOM 파싱
                current_product = None
                
                for idx, row in df_raw.iterrows():
                    # 제품명 행 (첫 번째 셀만 값이 있음)
                    if pd.notna(row[0]) and row[0] != '' and \
                       (pd.isna(row[1]) or row[1] == '') and \
                       (pd.isna(row[2]) or row[2] == ''):
                        current_product = row[0]
                        self.bom_data[current_product] = []
                    # 원료 행 (헤더 제외)
                    elif pd.notna(row[0]) and row[0] != '' and \
                         row[0] != 'ERP 코드' and current_product:
                        try:
                            self.bom_data[current_product].append({
                                '원료코드': int(float(row[0])) if row[0] else 0,
                                '원료명': row[1] if len(row) > 1 else '',
                                '배합률': float(row[2]) if len(row) > 2 and row[2] else 0.0
                            })
                        except:
                            continue
                
                # 자동 브랜드 매핑 생성
                self.brand_products = {'밥이보약': [], '더리얼': [], '기타': []}
                
                for product_name in self.bom_data.keys():
                    brand = self.detect_brand(product_name)
                    self.brand_products[brand].append(product_name)
                
                self.bom_available = len(self.bom_data) > 0
                
                if self.bom_available:
                    brand_summary = {
                        brand: len(products) 
                        for brand, products in self.brand_products.items()
                    }
                    st.success(
                        f"✅ BOM 데이터 로드 완료! (Google Sheets 연동)\n"
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
        found_in_products = []
        
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
                            found_in_products.append(product)
                            break
            
            if material_ratios:
                avg_ratio = np.mean(material_ratios) / 100
                requirement = brand_production * avg_ratio * 1000
                total_requirement += requirement
        
        return total_requirement if total_requirement > 0 else None
    
    def load_data(self, load_bom=True):
        """Google Sheets에서 데이터 로드 (개선 버전)"""
        try:
            with st.spinner("📊 데이터 로딩 중 (Google Sheets)..."):
                # 사용량 데이터
                st.write("📄 사용량 시트 로딩...")
                self.df_usage = self.sheets.read_sheet(SPREADSHEET_URLS['usage'], '사용량')
                if self.df_usage is None or len(self.df_usage) == 0:
                    st.error("❌ 사용량 시트가 비어있거나 읽을 수 없습니다.")
                    return False
                st.success(f"✅ 사용량: {len(self.df_usage)}개 원료")
                
                # 구매량 데이터
                st.write("📄 구매량 시트 로딩...")
                self.df_purchase = self.sheets.read_sheet(SPREADSHEET_URLS['usage'], '구매량')
                if self.df_purchase is None:
                    st.warning("⚠️ 구매량 시트를 읽을 수 없습니다.")
                    self.df_purchase = pd.DataFrame()
                else:
                    st.success(f"✅ 구매량: {len(self.df_purchase)}개 원료")
                
                # 월별 생산량
                st.write("📄 월별 생산량 시트 로딩...")
                self.df_production = self.sheets.read_sheet(SPREADSHEET_URLS['usage'], '월별 생산량')
                if self.df_production is None or len(self.df_production) == 0:
                    st.warning("⚠️ 월별 생산량 시트를 읽을 수 없습니다. 기본값을 사용합니다.")
                    self.df_production = pd.DataFrame()
                else:
                    st.success(f"✅ 월별 생산량: {len(self.df_production)}개 행")
                
                # 브랜드 비중
                st.write("📄 브랜드 비중 시트 로딩...")
                self.df_brand = self.sheets.read_sheet(SPREADSHEET_URLS['usage'], '브랜드 비중')
                if self.df_brand is None or len(self.df_brand) == 0:
                    st.warning("⚠️ 브랜드 비중 시트를 읽을 수 없습니다. 기본값을 사용합니다.")
                    self.df_brand = pd.DataFrame()
                else:
                    st.success(f"✅ 브랜드 비중: {len(self.df_brand)}개 브랜드")
                
                # 재고 데이터
                st.write("📄 재고현황 시트 로딩...")
                self.df_inventory = self.sheets.read_sheet(SPREADSHEET_URLS['inventory'], '재고현황')
                if self.df_inventory is None:
                    st.warning("⚠️ 재고현황 시트를 읽을 수 없습니다.")
                    self.df_inventory = pd.DataFrame()
                else:
                    st.success(f"✅ 재고현황: {len(self.df_inventory)}개 품목")
                
                # 데이터 타입 변환
                st.write("🔄 데이터 타입 변환 중...")
                self.convert_data_types()
            
            # 시계열 데이터 준비
            st.write("📈 시계열 데이터 준비 중...")
            self.prepare_time_series()
            
            # BOM 로드 (선택적)
            if load_bom:
                self.load_bom_data()
            
            st.success("✅ 모든 데이터 로드 완료!")
            return True
            
        except Exception as e:
            st.error(f"❌ 데이터 로드 실패: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False
    
    def convert_data_types(self):
        """데이터 타입 변환 (개선 버전 - 안전하게)"""
        try:
            # 사용량 데이터의 월 컬럼을 숫자로 변환
            month_cols = [col for col in self.df_usage.columns if '월' in col]
            
            for col in month_cols:
                if col in self.df_usage.columns:
                    # 각 셀에 safe_float 적용
                    self.df_usage[col] = self.df_usage[col].apply(self.safe_float)
                
                if col in self.df_purchase.columns:
                    self.df_purchase[col] = self.df_purchase[col].apply(self.safe_float)
            
            # 원료코드 변환
            if '원료코드' in self.df_usage.columns:
                self.df_usage['원료코드'] = self.df_usage['원료코드'].apply(
                    lambda x: int(self.safe_float(x)) if self.safe_float(x) > 0 else 0
                )
            
            # 재고 데이터 변환
            if len(self.df_inventory) > 0 and '품목코드' in self.df_inventory.columns:
                self.df_inventory['품목코드'] = self.df_inventory['품목코드'].apply(
                    lambda x: int(self.safe_float(x)) if self.safe_float(x) > 0 else 0
                )
                
                for col in self.df_inventory.columns:
                    if col not in ['품목코드', '품목명']:
                        self.df_inventory[col] = self.df_inventory[col].apply(self.safe_float)
            
            st.success("✅ 데이터 타입 변환 완료!")
            
        except Exception as e:
            st.warning(f"⚠️ 데이터 타입 변환 중 경고: {str(e)}")
            pass
    
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
        """시계열 데이터 준비 (개선 버전 - 특수 형식 처리)"""
        try:
            self.available_months = self.detect_month_columns(self.df_usage)
            num_months = len(self.available_months)
            
            if num_months == 0:
                st.error("❌ 월 데이터를 찾을 수 없습니다.")
                return
            
            self.months = pd.date_range(start='2025-01-01', periods=num_months, freq='MS')
            self.num_months = num_months
            
            # 생산량 데이터 준비 (개선 버전)
            default_prod = [345, 430, 554, 570, 522, 556, 606, 539, 580, 600, 620, 550]
            production_values = []
            
            if len(self.df_production) > 0:
                production_row = self.df_production.iloc[0]
                
                for i, col in enumerate(self.available_months):
                    if col in self.df_production.columns:
                        # safe_float가 "톤"을 제거해줌
                        val = self.safe_float(production_row[col])
                        if val > 0:
                            production_values.append(val)
                        else:
                            production_values.append(default_prod[min(i, len(default_prod)-1)])
                    else:
                        production_values.append(default_prod[min(i, len(default_prod)-1)])
            
            # 기본값으로 채우기
            if len(production_values) == 0:
                production_values = default_prod[:num_months]
            
            while len(production_values) < num_months:
                production_values.append(default_prod[min(len(production_values), len(default_prod)-1)])
            
            production_values = production_values[:num_months]
            
            self.production_ts = pd.DataFrame({
                'ds': self.months,
                'y': production_values
            })
            
            st.success(f"✅ 생산량 데이터: {production_values[:3]}... (톤)")
            
            # 브랜드 비중 (개선 버전 - % 처리)
            self.brand_ratios = {}
            default_ratios = {'밥이보약': 0.65, '더리얼': 0.33, '기타': 0.02}
            
            for brand in ['밥이보약', '더리얼', '기타']:
                ratios = []
                
                try:
                    # 브랜드명으로 행 찾기 (더 안전하게)
                    brand_row = None
                    
                    for idx, row in self.df_brand.iterrows():
                        first_col = str(row.iloc[0]).strip()
                        if first_col == brand:
                            brand_row = row
                            break
                    
                    if brand_row is not None:
                        for col in self.available_months:
                            if col in self.df_brand.columns:
                                # safe_float가 "%"를 제거해줌
                                val = self.safe_float(brand_row[col])
                                
                                # 값이 있으면 사용
                                if val > 0:
                                    # 이미 비율(0~1)인지 퍼센트(0~100)인지 확인
                                    if val > 1:
                                        ratios.append(val / 100)
                                    else:
                                        ratios.append(val)
                                else:
                                    ratios.append(default_ratios[brand])
                            else:
                                ratios.append(default_ratios[brand])
                    else:
                        st.warning(f"⚠️ '{brand}' 브랜드를 찾을 수 없어 기본값 사용")
                        
                except Exception as e:
                    st.warning(f"⚠️ '{brand}' 비중 로드 실패: {str(e)}, 기본값 사용")
                
                # 기본값으로 채우기
                if len(ratios) == 0:
                    ratios = [default_ratios[brand]] * num_months
                
                while len(ratios) < num_months:
                    ratios.append(default_ratios[brand])
                
                self.brand_ratios[brand] = ratios[:num_months]
            
            # 브랜드 비중 확인 메시지
            st.success(f"✅ 브랜드 비중 (첫 달): 밥이보약 {self.brand_ratios['밥이보약'][0]*100:.0f}%, 더리얼 {self.brand_ratios['더리얼'][0]*100:.0f}%, 기타 {self.brand_ratios['기타'][0]*100:.0f}%")
                
        except Exception as e:
            st.error(f"❌ 시계열 데이터 준비 실패: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            raise
    
    def safe_float(self, val):
        """안전한 float 변환 (개선 버전 - 특수문자 처리)"""
        try:
            if pd.isna(val) or val is None or val == '':
                return 0.0
            
            # 문자열로 변환
            if not isinstance(val, str):
                val = str(val)
            
            # 특수문자 제거: 톤, %, 쉼표, 공백 등
            val = val.replace('톤', '').replace('%', '').replace(',', '').strip()
            
            # 빈 문자열이면 0 반환
            if val == '':
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
        """개별 원료 예측"""
        try:
            if sum(usage_values) == 0:
                return 0, (0, 0), 'N/A'
            
            cleaned = self.remove_outliers(usage_values)
            material_type = self.classify_material(cleaned)
            weights = self.hybrid_weights[material_type]
            
            avg_prod = np.mean(self.production_ts['y'].values)
            prod_ratio = next_month_production / avg_prod if avg_prod > 0 else 1
            
            historical_max = max(cleaned) if cleaned else 0
            historical_avg = np.mean(cleaned) if cleaned else 0
            
            # BOM 예측
            bom_pred = self.calculate_bom_requirement(material_code, next_month_production, brand_ratios)
            
            bom_safe = False
            if bom_pred is not None and bom_pred > 0:
                if historical_max > 0 and bom_pred > historical_max * 2:
                    bom_pred = None
                    bom_safe = False
                else:
                    bom_safe = True
            
            # Prophet 예측
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
            
            # 트렌드 예측
            trend_pred = self.calculate_trend(cleaned) * prod_ratio
            
            # 이동평균
            ma_pred = np.mean(cleaned[-3:]) * prod_ratio
            
            # 하이브리드 앙상블
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
            
            # 보정
            if material_code in self.material_corrections:
                final_pred *= self.material_corrections[material_code]
            
            # 브랜드 보정
            if '닭' in str(material_name) or 'MDCM' in str(material_name):
                final_pred *= (1 + (brand_ratios['밥이보약'] - 0.62) * 0.2)
            elif '소고기' in str(material_name) or '연어' in str(material_name):
                final_pred *= (1 + (brand_ratios['더리얼'] - 0.35) * 0.3)
            
            # 신뢰구간
            margin = weights['base_margin']
            lower = final_pred * (1 - margin)
            upper = final_pred * (1 + margin)
            
            return final_pred, (lower, upper), confidence
            
        except:
            return np.mean(usage_values[-3:]) if usage_values else 0, (0, 0), 'N/A'
    
    def get_inventory(self, material_code):
        """재고 조회 (개선 버전)"""
        try:
            if len(self.df_inventory) == 0:
                return 0
            
            # 원료코드를 문자열로 변환하여 비교
            material_code_str = str(int(material_code)) if material_code > 0 else str(material_code)
            
            # 품목코드 컬럼에서 매칭
            for idx, row in self.df_inventory.iterrows():
                row_code = str(int(self.safe_float(row['품목코드']))) if '품목코드' in row.index else ''
                
                if row_code == material_code_str:
                    # 마지막 유효한 재고값 찾기 (역순으로)
                    for col in reversed(self.df_inventory.columns):
                        if col not in ['품목코드', '품목명'] and col != '':
                            val = self.safe_float(row[col])
                            if val > 0:
                                return val
                    
                    # 모든 컬럼 체크 (Unnamed 포함)
                    for col in reversed(self.df_inventory.columns):
                        if col not in ['품목코드', '품목명']:
                            val = self.safe_float(row[col])
                            if val > 0:
                                return val
        except Exception as e:
            pass
        
        return 0
    
    def predict_all(self, next_month_production, brand_ratios):
        """전체 예측 (개선 버전)"""
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
            
            if idx > 0 and idx % 10 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (idx + 1) * (total - idx - 1)
                time_text.text(f'예상 남은 시간: {eta:.0f}초')
            
            try:
                # 원료코드 안전하게 변환
                material_code = int(self.safe_float(row['원료코드']))
                if material_code == 0:
                    continue
            except:
                continue
                
            material_name = str(row['품목명']) if '품목명' in row.index else 'Unknown'
            
            usage_values = []
            for col in self.available_months:
                if col in row.index:
                    usage_values.append(self.safe_float(row[col]))
            
            # 모든 값이 0이면 스킵
            if sum(usage_values) == 0:
                continue
            
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
    # 1. 원료 분류 분포
    fig_pie = px.pie(
        df['원료_분류'].value_counts().reset_index(),
        values='count',
        names='원료_분류',
        title="원료 분류별 분포",
        color_discrete_map={'대량': '#1f77b4', '중간': '#ff7f0e', '소량': '#2ca02c'}
    )
    
    # 2. TOP 10 사용량
    top10 = df.nlargest(10, '예측_사용량')
    fig_bar = px.bar(
        top10,
        x='예측_사용량',
        y='품목명',
        orientation='h',
        title="TOP 10 예측 사용량",
        color='원료_분류',
        text='예측_사용량'
    )
    fig_bar.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    # 3. 예측 방식 분포
    if '예측_방식' in df.columns:
        fig_method = px.pie(
            df['예측_방식'].value_counts().reset_index(),
            values='count',
            names='예측_방식',
            title="예측 방식 분포",
            color_discrete_map={
                'BOM+AI': '#28a753', 
                'AI only': '#ffc107',
                'AI (BOM차단)': '#dc3545'
            }
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
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="예측결과_v7.1_sheets.xlsx">📥 엑셀 다운로드</a>'

def main():
    """메인 애플리케이션"""
    
    # 헤더
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎯 원료 예측 시스템 v7.1")
        st.markdown("**Google Sheets 연동 버전** (Prophet 65% + BOM 15%)")
    with col2:
        st.markdown("""
        <div class="success-box">
        <span class="google-sheets-badge">Google Sheets</span><br>
        <b>v7.1 특징</b><br>
        • 🛡️ 안전장치<br>
        • ☁️ 클라우드 연동<br>
        • 📊 실시간 데이터
        </div>
        """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # Google Sheets 인증
        st.subheader("🔐 Google Sheets 인증")
        
        credentials_file = st.file_uploader(
            "서비스 계정 JSON 키 파일",
            type=['json'],
            help="Google Cloud Console에서 생성한 서비스 계정 JSON 키"
        )
        
        if credentials_file:
            # 연결 상태 확인
            if 'sheets_connector' not in st.session_state:
                st.session_state.sheets_connector = GoogleSheetsConnector()
            
            sheets = st.session_state.sheets_connector
            
            if not sheets.connected:
                credentials_content = credentials_file.read()
                if sheets.connect(credentials_content):
                    st.success("✅ Google Sheets 연결 성공!")
                else:
                    st.error("❌ 연결 실패")
                    return
            else:
                st.success("✅ Google Sheets 연결됨")
            
            # 캐시 초기화 버튼
            if st.button("🔄 캐시 초기화", help="데이터를 새로 불러오려면 클릭"):
                sheets.clear_cache()
                st.cache_data.clear()
        else:
            st.info("💡 JSON 키 파일을 업로드하세요")
            
            with st.expander("📝 JSON 키 생성 방법"):
                st.markdown("""
                1. [Google Cloud Console](https://console.cloud.google.com/) 접속
                2. 프로젝트 생성 또는 선택
                3. **API 및 서비스 > 사용 설정된 API 및 서비스**
                4. "**Google Sheets API**" 및 "**Google Drive API**" 검색 후 사용 설정
                5. **API 및 서비스 > 사용자 인증 정보**
                6. **사용자 인증 정보 만들기 > 서비스 계정**
                7. 서비스 계정 생성 후 **키 추가 > JSON** 선택
                8. 다운로드된 JSON 파일을 여기에 업로드
                
                **중요:** 서비스 계정 이메일에 스프레드시트 편집 권한 부여!
                """)
            
            # API 할당량 정보
            with st.expander("⚠️ API 할당량 정보"):
                st.markdown("""
                **Google Sheets API 제한:**
                - 📊 분당 읽기 요청: 60-100개
                - ⏰ 할당량 리셋: 매 분마다
                
                **429 에러 발생 시:**
                1. ⏰ 1-2분 기다리기
                2. 🔄 캐시 초기화 버튼 사용하지 말기
                3. 💾 캐시 활용으로 API 호출 최소화
                
                **캐싱 기능:**
                - ✅ 한 번 읽은 데이터는 5분간 캐시
                - ✅ 새로고침해도 API 호출 안 함
                - ✅ "💾 캐시에서 로드" 메시지 확인
                """)
            return
        
        st.markdown("---")
        
        # BOM 사용 여부
        use_bom = st.checkbox("BOM 데이터 사용", value=True, help="참고용으로 활용 (15%)")
        
        st.markdown("---")
        
        # 예측 조건
        st.subheader("📝 예측 조건")
        
        production = st.number_input(
            "생산 계획 (톤)",
            min_value=100.0,
            max_value=1000.0,
            value=600.0,
            step=10.0
        )
        
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
        
        brand_ratios = {
            '밥이보약': bob/100,
            '더리얼': real/100,
            '기타': etc/100
        }
        
        st.markdown("---")
        
        # 실행 버튼
        predict_btn = st.button(
            "🔮 예측 실행",
            type="primary",
            use_container_width=True,
            disabled=(credentials_file is None)
        )
        
        # 스프레드시트 정보
        with st.expander("📊 연동된 스프레드시트"):
            st.markdown("""
            **사용량/구매량 데이터**
            - ID: `1lBanCoyOxv71LmXT316mO4XRccMyv5ETKcTcvm8wfvI`
            - 시트: 사용량, 구매량, 월별 생산량, 브랜드 비중
            
            **재고 데이터**
            - ID: `1k0_QxRBetfP8dFhHH5J478aFPvoMDvn_OPj1428CAzw`
            - 시트: 재고현황
            
            **BOM 데이터**
            - ID: `1vdkYQ9tQzuj_juXZPhgEsDdhAXGWqtCejXLZHXNsAws`
            - 시트: 제품 BOM
            """)
        
        # 모델 정보
        with st.expander("📊 모델 정보"):
            st.markdown("""
            **v7.1 하이브리드 구성**
            
            BOM 안전할 때:
            - Prophet: 60-65% ⭐
            - BOM: 10-15%
            - 트렌드: 15-20%
            - 이동평균: 5-10%
            
            BOM 불안전할 때:
            - Prophet: 73%
            - 트렌드: 18%
            - 이동평균: 9%
            - BOM: 차단! 🛡️
            
            **Google Sheets 연동 장점**
            - ☁️ 실시간 데이터 동기화
            - 👥 팀 협업 용이
            - 📱 어디서나 접근 가능
            - 🔄 자동 백업
            """)
    
    # 메인 영역
    if credentials_file and 'sheets_connector' in st.session_state:
        sheets = st.session_state.sheets_connector
        
        if sheets.connected:
            # 모델 초기화
            if 'model' not in st.session_state or st.session_state.get('model_sheets') != sheets:
                st.session_state.model = BOMHybridModel(sheets)
                st.session_state.model_sheets = sheets
            
            model = st.session_state.model
            
            # 데이터 로드
            if model.load_data(load_bom=use_bom):
                # 정보 표시
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("원료 수", f"{len(model.df_usage):,}")
                with col2:
                    st.metric("데이터 기간", f"1-{model.num_months}월")
                with col3:
                    st.metric("생산 계획", f"{production:.0f}톤")
                with col4:
                    if model.bom_available:
                        st.metric("BOM 제품", f"{len(model.bom_data)}개", delta="통합됨", delta_color="normal")
                    else:
                        st.metric("BOM 상태", "미사용", delta="기존방식", delta_color="off")
                
                if predict_btn:
                    st.markdown("---")
                    st.header("📈 예측 결과")
                    
                    # 예측 실행
                    with st.container():
                        predictions = model.predict_all(production, brand_ratios)
                    
                    if predictions is not None and not predictions.empty:
                        # 요약
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("총 예측 사용량", f"{predictions['예측_사용량'].sum():,.0f}")
                        with col2:
                            st.metric("총 예측 구매량", f"{predictions['예측_구매량'].sum():,.0f}")
                        with col3:
                            avg_range = predictions['신뢰구간_폭'].apply(
                                lambda x: float(x.replace('±', '').replace('%', ''))
                            ).mean()
                            st.metric("평균 신뢰구간", f"±{avg_range:.1f}%")
                        with col4:
                            if model.bom_available:
                                bom_count = len(predictions[predictions['예측_방식']=='BOM+AI'])
                                st.metric("BOM 적용", f"{bom_count}개", delta=f"{bom_count/len(predictions)*100:.0f}%")
                            else:
                                st.metric("예측 방식", "AI only")
                        
                        # 탭
                        tab1, tab2, tab3, tab4 = st.tabs(
                            ["📊 차트", "📋 데이터", "🎯 TOP 20", "📥 다운로드"]
                        )
                        
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
                            # 필터
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                categories = st.multiselect(
                                    "분류 필터",
                                    ['대량', '중간', '소량'],
                                    ['대량', '중간', '소량']
                                )
                            with col2:
                                if model.bom_available:
                                    methods = st.multiselect(
                                        "예측 방식",
                                        ['BOM+AI', 'AI only', 'AI (BOM차단)'],
                                        ['BOM+AI', 'AI only', 'AI (BOM차단)']
                                    )
                                else:
                                    methods = ['AI only']
                            with col3:
                                search = st.text_input("원료명 검색")
                            
                            # 필터링
                            filtered = predictions[predictions['원료_분류'].isin(categories)]
                            if model.bom_available:
                                filtered = filtered[filtered['예측_방식'].isin(methods)]
                            if search:
                                filtered = filtered[
                                    filtered['품목명'].str.contains(search, case=False, na=False)
                                ]
                            
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
                            
                            # 엑셀 다운로드
                            st.markdown(get_download_link(predictions), unsafe_allow_html=True)
                            
                            # CSV 다운로드
                            csv = predictions.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                "📄 CSV 다운로드",
                                csv,
                                "predictions_v7.1_sheets.csv",
                                "text/csv"
                            )
                            
                            # 요약 정보
                            bom_status = f"BOM 통합 ({len(model.bom_data)}개 제품)" if model.bom_available else "BOM 미사용"
                            blocked_count = len(predictions[predictions['예측_방식']=='AI (BOM차단)']) if model.bom_available else 0
                            st.info(f"""
                            **파일 정보**
                            - 원료: {len(predictions)}개
                            - 데이터 기간: 1-{model.num_months}월
                            - 모델: v7.1 하이브리드 (Google Sheets)
                            - BOM: {bom_status}
                            - 안전장치 작동: {blocked_count}개 원료
                            - 평균 신뢰구간: ±{avg_range:.1f}%
                            - 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                            """)
    else:
        # 초기 화면
        st.info("👈 좌측 사이드바에서 Google Sheets 인증을 진행하세요")
        
        # 중요 공지
        st.warning("""
        ⚠️ **API 할당량 안내**
        
        Google Sheets API는 **분당 60-100개 읽기 요청 제한**이 있어요.
        
        **429 에러 발생 시:**
        - ⏰ 1-2분만 기다렸다가 다시 실행하세요
        - 💾 캐싱 기능이 자동으로 API 호출을 최소화합니다
        - 🔄 불필요한 캐시 초기화는 피해주세요
        """)
        
        with st.expander("🚀 Google Sheets 연동 버전", expanded=True):
            st.markdown("""
            ### ☁️ 클라우드 기반 예측 시스템
            
            **Google Sheets 연동 장점**
            - 📊 실시간 데이터 동기화
            - 👥 팀원과 함께 데이터 관리
            - 📱 언제 어디서나 접근 가능
            - 🔄 자동 버전 관리 및 백업
            - 💾 파일 업로드 불필요
            - 💾 **캐싱으로 빠른 재실행**
            
            **사용 방법**
            1. Google Cloud Console에서 서비스 계정 생성
            2. Google Sheets API, Drive API 활성화
            3. JSON 키 파일 다운로드
            4. 스프레드시트에 서비스 계정 이메일 공유 권한 부여
            5. JSON 키 파일 업로드
            6. 예측 실행!
            
            **연동된 스프레드시트**
            - ✅ 사용량 및 구매량 예측모델
            - ✅ 월별 기초재고 및 기말재고
            - ✅ BOM 신뢰성 추가 (선택)
            
            **v7.1 핵심 기능**
            - 🛡️ BOM 안전장치 (과대예측 방지)
            - 📊 Prophet 65% 강화
            - 🤖 자동 브랜드 인식
            - 🎯 3가지 예측 방식
            - 💾 **스마트 캐싱 (API 할당량 절약)**
            """)
        
        st.success("""
        💡 **시작하기**
        1. 왼쪽 사이드바에서 JSON 키 파일 업로드
        2. Google Sheets 연결 확인
        3. 생산 계획 및 브랜드 비중 입력
        4. 🔮 예측 실행 버튼 클릭!
        
        **💾 캐싱 기능**
        - 한 번 읽은 데이터는 5분간 자동 저장
        - 새로고침해도 다시 읽지 않음
        - API 할당량 걱정 없이 사용!
        """)

if __name__ == "__main__":
    main()



