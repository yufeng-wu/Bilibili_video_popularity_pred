'''
Defines a class BiliSpider, which is a worker that scrapes video stats from Bilibili.
'''

import requests
import random
import pprint
from time import sleep

class BiliSpider:

    def __init__(self):
        self.get_video_stat_url = 'http://api.bilibili.com/x/web-interface/view/detail'
        self.get_video_basic_stat_url = 'http://api.bilibili.com/x/web-interface/view'

        self.user_agent=[
            "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
            "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
            "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
            "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
            "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
            "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
            "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
            "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
            "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
            "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
            "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
        ]

        self.proxy = [
            {
            'HTTP':'202.108.22.5:80',
            'HTTPS':'202.108.22.5:80'
            },
            {
            'HTTP':'60.255.151.82:80',
            'HTTPS':'60.255.151.82:80'
            },
            {
            'HTTP':'60.255.151.81:80',
            'HTTPS':'60.255.151.81:80'
            },
            {
            'HTTP':'58.240.52.114:80',
            'HTTPS':'58.240.52.114:80'
            },
            {
            'HTTP':'202.108.22.5:80',
            'HTTPS':'202.108.22.5:80'
            },
            {
            'HTTP':'117.251.103.186:8080',
            'HTTPS':'117.251.103.186:8080'
            },
            {
            'HTTP':'208.82.61.38:3128',
            'HTTPS':'208.82.61.38:3128'
            },
            {
            'HTTP':'64.29.86.251:3129',
            'HTTPS':'64.29.86.251:3129'
            },
            {
            'HTTP':'155.4.244.218:80',
            'HTTPS':'155.4.244.218:80'
            },
            {
            'HTTP':'208.82.61.13:3128',
            'HTTPS':'208.82.61.13:3128'
            },
        ]

    def _get_api(self, api_url, params=None):
        trail = 0
        MAX_TRIALS = 5

        while trail < MAX_TRIALS:
            headers = {"User-Agent": random.choice(self.user_agent)}
            try:
                trail += 1
                res = requests.get(api_url, params=params, headers=headers, proxies=random.choice(self.proxy), timeout=10)
                break
            except requests.exceptions.ConnectionError:
                print("requests.exceptions.ConnectionError ERROR occured. Will sleep for 5 - 10 sec")
                sleep(random.randint(5,10) * trail)

        return res.json()

    def get_video_stat(self, bvid):
        '''
        bvid: BV number of a video
        return: a dictionary containing video stats
        '''
        params = {
            'bvid': bvid,
        }

        return self._get_api(self.get_video_stat_url, params)

    def get_basic_video_stat(self, bvid):
        '''
        bvid: BV number of a video
        return: a dictionary containing basic video stats
        '''
        params = {
            'bvid': bvid,
        }

        return self._get_api(self.get_video_basic_stat_url, params)


if __name__ == '__main__':
    # Test the BiliSpider class
    bili = BiliSpider()
    pp = pprint.PrettyPrinter(indent=4)
    bvid = 'BV1od4y147Tr'
    pp.pprint(bili.get_basic_video_stat(bvid)['data']['desc'])
