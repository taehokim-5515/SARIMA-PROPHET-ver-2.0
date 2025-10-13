"""
디버깅 버전 - Google Sheets 연동 테스트
문제 발생 지점을 정확히 파악하기 위한 상세 로깅

실행: streamlit run app_debug.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import json
import traceback

warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="🔍 디버깅 모드",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Google Sheets 연동 디버깅 모드")
st.markdown("각 단계별로 상세한 정보를 확인할 수 있어요!")

# 스프레드시트 URL
SPREADSHEET_URLS = {
    'bom': '1vdkYQ9tQzuj_juXZPhgEsDdhAXGWqtCejXLZHXNsAws',
    'usage': '1lBanCoyOxv71LmXT316mO4XRccMyv5ETKcTcvm8wfvI',
    'inventory': '1k0_QxRBetfP8dFhHH5J478aFPvoMDvn_OPj1428CAzw'
}

def log_info(message, data=None, level="info"):
    """로그 출력 헬퍼"""
    if level == "info":
        st.info(f"ℹ️ {message}")
    elif level == "success":
        st.success(f"✅ {message}")
    elif level == "warning":
        st.warning(f"⚠️ {message}")
    elif level == "error":
        st.error(f"❌ {message}")
    
    if data is not None:
        with st.expander("📊 데이터 상세보기"):
            st.write(data)

def test_google_sheets_connection(credentials_json):
    """Google Sheets 연결 테스트"""
    st.header("1️⃣ Google Sheets 연결 테스트")
    
    try:
        # JSON 파싱
        log_info("JSON 키 파싱 중...")
        if isinstance(credentials_json, bytes):
            credentials_dict = json.loads(credentials_json.decode('utf-8'))
        else:
            credentials_dict = json.loads(credentials_json)
        
        log_info("✅ JSON 키 파싱 성공", level="success")
        
        # 서비스 계정 이메일 표시
        service_email = credentials_dict.get('client_email', 'N/A')
        st.info(f"📧 서비스 계정: `{service_email}`")
        st.warning("⚠️ 이 이메일이 스프레드시트에 공유되어 있는지 확인하세요!")
        
        # 인증 범위 설정
        log_info("인증 범위 설정 중...")
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly'
        ]
        
        # 자격증명 생성
        log_info("자격증명 생성 중...")
        creds = Credentials.from_service_account_info(
            credentials_dict,
            scopes=scopes
        )
        
        # gspread 클라이언트 생성
        log_info("gspread 클라이언트 생성 중...")
        client = gspread.authorize(creds)
        
        log_info("✅ Google Sheets 연결 성공!", level="success")
        
        return client
        
    except Exception as e:
        log_info(f"연결 실패: {str(e)}", level="error")
        st.code(traceback.format_exc())
        return None

def test_read_spreadsheet(client, spreadsheet_id, spreadsheet_name):
    """개별 스프레드시트 읽기 테스트"""
    st.header(f"2️⃣ 스프레드시트 읽기 테스트: {spreadsheet_name}")
    
    try:
        log_info(f"스프레드시트 ID: `{spreadsheet_id}` 열기 중...")
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        log_info(f"✅ 스프레드시트 열기 성공: '{spreadsheet.title}'", level="success")
        
        # 모든 시트 목록 표시
        worksheets = spreadsheet.worksheets()
        sheet_titles = [ws.title for ws in worksheets]
        
        st.success(f"📄 발견된 시트 목록 ({len(sheet_titles)}개):")
        for idx, title in enumerate(sheet_titles, 1):
            st.write(f"  {idx}. `{title}`")
        
        return spreadsheet, sheet_titles
        
    except Exception as e:
        log_info(f"스프레드시트 열기 실패: {str(e)}", level="error")
        st.code(traceback.format_exc())
        return None, []

def test_read_worksheet(spreadsheet, sheet_name):
    """개별 워크시트 읽기 테스트"""
    st.subheader(f"📋 시트 읽기: '{sheet_name}'")
    
    try:
        log_info(f"시트 '{sheet_name}' 선택 중...")
        worksheet = spreadsheet.worksheet(sheet_name)
        
        log_info(f"✅ 시트 선택 성공", level="success")
        
        # 시트 정보
        st.write(f"- 행 수: {worksheet.row_count}")
        st.write(f"- 열 수: {worksheet.col_count}")
        
        # 데이터 읽기
        log_info("데이터 읽기 중...")
        data = worksheet.get_all_values()
        
        if not data:
            log_info("⚠️ 시트가 비어있습니다!", level="warning")
            return None
        
        log_info(f"✅ 데이터 읽기 성공: {len(data)}행", level="success")
        
        # DataFrame 변환
        log_info("DataFrame 변환 중...")
        df = pd.DataFrame(data[1:], columns=data[0])
        
        st.success(f"📊 DataFrame 생성 완료!")
        st.write(f"- Shape: {df.shape}")
        st.write(f"- Columns: {list(df.columns)}")
        
        # 데이터 미리보기
        with st.expander("🔍 데이터 미리보기 (처음 10행)"):
            st.dataframe(df.head(10))
        
        # 데이터 타입 확인
        with st.expander("📋 컬럼별 데이터 타입"):
            st.write(df.dtypes)
        
        # 통계 정보
        with st.expander("📈 기본 통계"):
            st.write(df.describe(include='all'))
        
        return df
        
    except gspread.exceptions.WorksheetNotFound:
        log_info(f"시트 '{sheet_name}'을 찾을 수 없습니다!", level="error")
        return None
    except Exception as e:
        log_info(f"시트 읽기 실패: {str(e)}", level="error")
        st.code(traceback.format_exc())
        return None

def test_data_processing(df, sheet_name):
    """데이터 처리 테스트"""
    st.subheader(f"🔧 데이터 처리 테스트: '{sheet_name}'")
    
    if df is None or len(df) == 0:
        log_info("DataFrame이 비어있어서 처리할 수 없습니다.", level="warning")
        return
    
    try:
        # 월 컬럼 찾기
        log_info("월 컬럼 찾기...")
        month_cols = [col for col in df.columns if '월' in col]
        st.write(f"📅 발견된 월 컬럼: {month_cols}")
        
        if month_cols:
            # 첫 번째 월 컬럼 샘플 데이터
            sample_col = month_cols[0]
            st.write(f"🔍 '{sample_col}' 컬럼 샘플 (처음 5개):")
            st.write(df[sample_col].head())
            
            # 숫자 변환 테스트
            log_info(f"'{sample_col}' 숫자 변환 테스트...")
            converted = pd.to_numeric(df[sample_col], errors='coerce')
            
            st.write(f"✅ 변환 결과:")
            st.write(f"- 변환 성공: {converted.notna().sum()}개")
            st.write(f"- 변환 실패 (NaN): {converted.isna().sum()}개")
            st.write(f"- 0이 아닌 값: {(converted > 0).sum()}개")
            
            with st.expander("변환된 데이터 샘플"):
                st.write(pd.DataFrame({
                    '원본': df[sample_col].head(10),
                    '변환후': converted.head(10)
                }))
        
        # 특정 컬럼 찾기
        log_info("주요 컬럼 확인...")
        key_columns = ['원료코드', '품목코드', '품목명', '원료명']
        found_columns = [col for col in key_columns if col in df.columns]
        st.write(f"✅ 발견된 주요 컬럼: {found_columns}")
        
        if found_columns:
            sample_col = found_columns[0]
            st.write(f"🔍 '{sample_col}' 샘플:")
            st.write(df[sample_col].head(10))
        
        # 빈 셀 확인
        log_info("빈 셀 확인...")
        total_cells = df.shape[0] * df.shape[1]
        empty_cells = (df == '').sum().sum()
        null_cells = df.isna().sum().sum()
        
        st.write(f"📊 데이터 품질:")
        st.write(f"- 전체 셀: {total_cells:,}개")
        st.write(f"- 빈 문자열: {empty_cells:,}개 ({empty_cells/total_cells*100:.1f}%)")
        st.write(f"- NULL: {null_cells:,}개 ({null_cells/total_cells*100:.1f}%)")
        
    except Exception as e:
        log_info(f"데이터 처리 테스트 실패: {str(e)}", level="error")
        st.code(traceback.format_exc())

def test_brand_ratio_processing(df_brand):
    """브랜드 비중 데이터 처리 테스트"""
    st.subheader("🏷️ 브랜드 비중 특별 테스트")
    
    if df_brand is None or len(df_brand) == 0:
        log_info("브랜드 비중 데이터가 없습니다.", level="warning")
        return
    
    try:
        log_info("브랜드 이름 찾기...")
        brands = ['밥이보약', '더리얼', '기타']
        
        st.write("🔍 첫 번째 컬럼 (브랜드명 예상):")
        st.write(df_brand.iloc[:, 0].head(10))
        
        for brand in brands:
            st.write(f"\n**'{brand}' 브랜드 찾기:**")
            
            # 방법 1: 정확히 일치
            exact_match = df_brand[df_brand.iloc[:, 0] == brand]
            st.write(f"  - 정확히 일치: {len(exact_match)}개 행")
            
            # 방법 2: 문자열로 변환 후 일치
            str_match = df_brand[df_brand.iloc[:, 0].astype(str) == brand]
            st.write(f"  - 문자열 변환 후 일치: {len(str_match)}개 행")
            
            # 방법 3: 포함 여부
            contains_match = df_brand[df_brand.iloc[:, 0].astype(str).str.contains(brand, na=False)]
            st.write(f"  - 포함된 경우: {len(contains_match)}개 행")
            
            if len(str_match) > 0:
                st.success(f"✅ '{brand}' 발견!")
                with st.expander(f"'{brand}' 데이터 보기"):
                    st.write(str_match)
            else:
                st.warning(f"⚠️ '{brand}'를 찾을 수 없습니다!")
        
    except Exception as e:
        log_info(f"브랜드 비중 테스트 실패: {str(e)}", level="error")
        st.code(traceback.format_exc())

def main():
    """메인 디버깅 함수"""
    
    st.sidebar.header("🔐 인증")
    credentials_file = st.sidebar.file_uploader(
        "서비스 계정 JSON 키",
        type=['json']
    )
    
    if not credentials_file:
        st.info("👈 좌측에서 JSON 키 파일을 업로드하세요")
        return
    
    # 1단계: 연결 테스트
    credentials_content = credentials_file.read()
    client = test_google_sheets_connection(credentials_content)
    
    if not client:
        st.error("❌ Google Sheets 연결에 실패했습니다. 위의 오류를 확인하세요.")
        return
    
    st.markdown("---")
    
    # 2단계: 스프레드시트 선택
    st.sidebar.header("📊 테스트할 스프레드시트")
    test_target = st.sidebar.radio(
        "선택:",
        ["사용량/구매량", "재고현황", "BOM"]
    )
    
    # 선택에 따라 테스트
    if test_target == "사용량/구매량":
        spreadsheet, sheet_titles = test_read_spreadsheet(
            client, 
            SPREADSHEET_URLS['usage'],
            "사용량/구매량 예측모델"
        )
        
        if spreadsheet:
            st.markdown("---")
            
            # 각 시트 테스트
            required_sheets = ['사용량', '구매량', '월별 생산량', '브랜드 비중']
            
            for sheet_name in required_sheets:
                if sheet_name in sheet_titles:
                    df = test_read_worksheet(spreadsheet, sheet_name)
                    
                    if df is not None:
                        test_data_processing(df, sheet_name)
                        
                        # 브랜드 비중은 특별 테스트
                        if sheet_name == '브랜드 비중':
                            st.markdown("---")
                            test_brand_ratio_processing(df)
                    
                    st.markdown("---")
                else:
                    st.error(f"❌ 필수 시트 '{sheet_name}'을 찾을 수 없습니다!")
                    st.write(f"📋 사용 가능한 시트: {sheet_titles}")
    
    elif test_target == "재고현황":
        spreadsheet, sheet_titles = test_read_spreadsheet(
            client,
            SPREADSHEET_URLS['inventory'],
            "월별 기초재고 및 기말재고"
        )
        
        if spreadsheet and '재고현황' in sheet_titles:
            st.markdown("---")
            df = test_read_worksheet(spreadsheet, '재고현황')
            
            if df is not None:
                test_data_processing(df, '재고현황')
    
    elif test_target == "BOM":
        spreadsheet, sheet_titles = test_read_spreadsheet(
            client,
            SPREADSHEET_URLS['bom'],
            "BOM 신뢰성 추가"
        )
        
        if spreadsheet and '제품 BOM' in sheet_titles:
            st.markdown("---")
            df = test_read_worksheet(spreadsheet, '제품 BOM')
            
            if df is not None:
                st.subheader("🔧 BOM 구조 분석")
                
                st.write("🔍 BOM 데이터는 특별한 구조를 가지고 있어요:")
                st.write("- 제품명 행: 첫 번째 셀만 값이 있음")
                st.write("- 원료 행: ERP 코드, 원료명, 배합률")
                
                # 첫 20행 구조 분석
                with st.expander("처음 20행 구조 분석"):
                    for idx in range(min(20, len(df))):
                        row = df.iloc[idx]
                        filled_cells = sum(1 for val in row if val and val != '')
                        
                        if filled_cells == 1:
                            st.success(f"행 {idx+1}: 제품명 행 → '{row.iloc[0]}'")
                        elif filled_cells >= 3:
                            st.info(f"행 {idx+1}: 원료 행 → ERP:{row.iloc[0]}, 원료:{row.iloc[1]}, 배합률:{row.iloc[2]}")
                        else:
                            st.write(f"행 {idx+1}: {filled_cells}개 셀 채워짐")
    
    # 종합 요약
    st.markdown("---")
    st.header("📝 디버깅 요약")
    
    st.info("""
    **다음 정보를 확인하세요:**
    
    1. ✅ **서비스 계정 이메일**이 모든 스프레드시트에 공유되어 있나요?
    2. ✅ **시트 이름**이 정확히 일치하나요? (공백, 대소문자 주의)
    3. ✅ **데이터 형식**이 올바른가요? (숫자는 숫자로, 텍스트는 텍스트로)
    4. ✅ **빈 셀**이 너무 많지 않나요?
    5. ✅ **월 컬럼**이 "1월", "2월" 형식인가요?
    
    위에서 발견된 문제를 수정하고 다시 테스트하세요!
    """)

if __name__ == "__main__":
    main()
