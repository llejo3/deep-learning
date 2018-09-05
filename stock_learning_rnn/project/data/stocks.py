import pandas as pd
import os
import datetime
from utils.date_utils import DateUtils
from data.data_utils import DataUtils


class Stocks:
    """ 주식데이터  """

    def _get_naver_url(self, comp_code):
        """ 네이버 금융(http://finance.naver.com)에 넣어줌 """
        return 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=comp_code)

    def get_stock_naver_data(self, comp_code, start_date):
        """네이버 매일 주식정보를 가져온다."""
        url = self._get_naver_url(comp_code)
        df = pd.DataFrame()

        # 네이버 웹 크롤링
        page = 1
        bf_date = ''
        while True:
            pg_url = '{url}&page={page}'.format(url=url, page=page)
            page_data = pd.read_html(pg_url, header=0)[0]
            page_data = page_data.dropna()
            last_date = page_data.tail(1)['날짜'].to_string(index=False)
            if bf_date == last_date:
                break
            df = df.append(page_data, ignore_index=True)
            if start_date != '':
                if DateUtils.to_date(start_date) > DateUtils.to_date(last_date):
                    break
            if len(page_data) < 10:
                break
            page += 1
            bf_date = last_date

            # 필요 없는 날짜 제거
        if start_date != '':
            drop_cnt = 0
            df_len = len(df)
            for i in range(df_len):
                last_date = df.loc[df_len - i - 1, '날짜']
                if DateUtils.to_date(start_date) > DateUtils.to_date(last_date):
                    drop_cnt += 1
                else:
                    break
            if drop_cnt > 0:
                df = df[:-drop_cnt]

        # 정렬 및 컬럼명 변경
        if df.shape[0] != 0:
            df = df.sort_values(by='날짜')
            df.rename(columns={'날짜': 'date',
                               '종가': 'close',
                               '전일비': 'diff',
                               '시가': 'open',
                               '고가': 'high',
                               '저가': 'low',
                               '거래량': 'volume'}, inplace=True)
        return df

    def get_stock_data(self, comp_code):
        comp_code = DataUtils.to_string_corp_code(comp_code)
        file_path = './data/files/stocks/' + comp_code + '.csv'

        if os.path.isfile(file_path):
            stock_data = pd.read_csv(file_path)
            stock_data = stock_data[:-1]
            date_last = stock_data.tail(1)['date'].to_string(index=False)
            date_next = DateUtils.to_date(date_last) + datetime.timedelta(days=1)
            date_next = date_next.strftime("%Y-%m-%d")
            new_data = self.get_stock_naver_data(comp_code, date_next)
            if len(new_data) > 0:
                stock_data = stock_data.append(new_data, ignore_index=True)
                stock_data.to_csv(file_path, index=False)
        else:
            stock_data = self.get_stock_naver_data(comp_code, '')
            stock_data.to_csv(file_path, index=False)
        return stock_data
