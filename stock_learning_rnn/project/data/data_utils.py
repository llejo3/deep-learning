import pandas as pd


class DataUtils:
    """ 데이터 처리 관련 메소드"""

    @staticmethod
    def save_excel(df_data, file_path):
        """ excel로 저장한다. """
        writer = pd.ExcelWriter(file_path)
        df_data.to_excel(writer, 'Sheet1', index=False)
        writer.save()

    @staticmethod
    def to_string_corp_code(corp_code):
        """주식회사 코드를 문자열로 바꾼다."""
        return format(int(corp_code), "06d")

    @staticmethod
    def get_session_path(corp_code):
        """저장할 세션의 경로 및 파일명"""
        comp_code = DataUtils.to_string_corp_code(corp_code)
        return "./data/files/sessions/" + comp_code + ".ckpt"
