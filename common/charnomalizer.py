#!usr/bin/python
# -*- coding:utf-8 -*-

import re

class CharNormalizer(object):

    def __init__(self):
        self.charmap_dic = {}
        self.charmap_dic['آ'] = '[ﺁ]'
        self.charmap_dic['ا'] = '[ﺍﺎ]'
        self.charmap_dic['ب'] = '[ﺏﺐﺑﺒ]'
        self.charmap_dic['پ'] = '[ﭖﭗﭘﭙ]'
        self.charmap_dic['ت'] = '[ﺕﺖﺗﺘ]'
        self.charmap_dic['ث'] = '[ﺙﺚﺛﺜ]'
        self.charmap_dic['ج'] = '[ﺝﺞﺟﺠ]'
        self.charmap_dic['چ'] = '[ﭺﭻﭼﭽﭾﭿﮀﮁ]'
        self.charmap_dic['ح'] = '[ﺡﺢﺣﺤ]'
        self.charmap_dic['خ'] = '[ﺥﺦﺧﺨ]'
        self.charmap_dic['د'] = '[ﺩﺪ]'
        self.charmap_dic['ذ'] = '[ﺫﺬ]'
        self.charmap_dic['ر'] = '[٫ﺭﺮ]'
        self.charmap_dic['ز'] = '[ﺯﺰ]'
        self.charmap_dic['ژ'] = '[ﮊﮋ]'
        self.charmap_dic['س'] = '[ﺱﺲﺳﺴ]'
        self.charmap_dic['ش'] = '[ﺵﺶﺷﺸ]'
        self.charmap_dic['ص'] = '[ﺹﺺﺻﺼ]'
        self.charmap_dic['ض'] = '[ﺽﺾﺿﻀ]'
        self.charmap_dic['ط'] = '[ﻁﻂﻃﻄ]'
        self.charmap_dic['ظ'] = '[ﻅﻆﻇﻈ]'
        self.charmap_dic['ع'] = '[ﻉﻊﻋﻌ]'
        self.charmap_dic['غ'] = '[ﻍﻎﻏﻐ]'
        self.charmap_dic['ف'] = '[ﻑﻒﻓﻔ]'
        self.charmap_dic['ق'] = '[ﻕﻖﻗﻘ]'
        self.charmap_dic['ک'] = '[كﮎﮏﮐﮑﻙﻚﻛﻜ]'
        self.charmap_dic['گ'] = '[ﮒﮓﮔﮕ]'
        self.charmap_dic['ل'] = '[ﻝﻞﻟﻠ]'
        self.charmap_dic['م'] = '[ﻡﻢﻣﻤ]'
        self.charmap_dic['ن'] = '[ﻥﻦﻧﻨ]'
        self.charmap_dic['و'] = '[ﻭﻮ]'
        self.charmap_dic['ه'] = '[ﻩﻪﻫﻬ]'
        self.charmap_dic['ی'] = '[يﯽﯾﯿﯼىﻯﻰﻱﻲﻳﻴ]'
        self.charmap_dic[':'] = '[：]'
        self.charmap_dic['"'] = '[‘’“”]'
        self.charmap_dic['1'] = '[١۱]'
        self.charmap_dic['2'] = '[٢۲]'
        self.charmap_dic['3'] = '[٣۳]'
        self.charmap_dic['4'] = '[٤۴]'
        self.charmap_dic['5'] = '[٥۵]'
        self.charmap_dic['6'] = '[٦۶]'
        self.charmap_dic['7'] = '[٧۷]'
        self.charmap_dic['8'] = '[٨۸]'
        self.charmap_dic['9'] = '[٩۹]'
        self.charmap_dic['0'] = '[٠۰]'
        self.charmap_dic[''] = '[ًٌٍَُِّْٰٓٔ]'

    def normalize(self, string_in):
        string_out = str(string_in).lower()
        for (key, value) in self.charmap_dic.items():
            string_out = re.sub(value, key, string_out)
        return string_out

