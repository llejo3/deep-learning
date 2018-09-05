import pandas as pd
import os
from data.data_utils import DataUtils


class Corp:
    """ 주식회사 정보  """
    _FILE_PATH = './data/files/corps.xlsx'

    def _save_corps(self):
        """ 주식회사 정보를 가져와서 엑셀로 저장한다. """
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
        code_df = pd.read_html(url, header=0)[0]
        DataUtils.save_excel(code_df, self._FILE_PATH)

    def get_corp_code(self, corp_name):
        """ 엘셀을 불러와서 회사 코드를 가져온다. """
        corps = self.get_corps()
        corp_code = corps.query("회사명=='{}'".format(corp_name))['종목코드'].to_string(index=False)
        return format(int(corp_code), "06d")

    def get_corp_codes(self, to_listing_date=''):
        corp_codes = self.get_corps(to_listing_date, '종목코드')
        return corp_codes

    def get_corps(self, to_listing_date='', columns=''):
        if not os.path.isfile(self._FILE_PATH):
            self._save_corps()
        corps = pd.read_excel(self._FILE_PATH)
        if to_listing_date != '':
            corps = corps.query("상장일<='{}'".format(to_listing_date))
        if columns != '':
            corps = corps[columns]
        return corps
