from yahooquery import Screener

crypto_list = [{'label': 'BTC-USD', 'value':'BTC-USD'},
               {'label': 'ETH-USD', 'value':'ETH-USD'},
               {'label': 'USDT-USD', 'value':'USDT-USD'},
               {'label': 'BNB-USD', 'value':'BNB-USD'},
               {'label': 'XRP-USD', 'value':'XRP-USD'},
               {'label': 'ADA-USD', 'value':'ADA-USD'},
               {'label': 'SOL-USD', 'value':'SOL-USD'},
               {'label': 'DOGE-USD', 'value':'DOGE-USD'},
               {'label': 'DOT-USD', 'value':'DOT-USD'},
               {'label': 'MATIC-USD', 'value':'MATIC-USD'},
               {'label': 'DAI-USD', 'value':'DAI-USD'},
               {'label': 'SHIB-USD', 'value':'SHIB-USD'},
               {'label': 'WTRX-USD', 'value':'WTRX-USD'},
               {'label': 'HEX-USD', 'value':'HEX-USD'},
               {'label': 'TRX-USD', 'value':'TRX-USD'},
               {'label': 'AVAX-USD', 'value':'AVAX-USD'},
               {'label': 'STEHT-USD', 'value':'STEHT-USD'},
               {'label': 'WBTC-USD', 'value':'WBTC-USD'},
               {'label': 'ATOM-USD', 'value':'ATOM-USD'},
               {'label': 'LEO-USD', 'value':'LEO-USD'},
               {'label': 'ETC-USD', 'value':'ETC-USD'},
               {'label': 'UNI-USD', 'value':'UNI-USD'},
               {'label': 'YOUC-USD', 'value':'YOUC-USD'},
               {'label': 'LTC-USD', 'value':'LTC-USD'},
               {'label': 'NEAR-USD', 'value':'NEAR-USD'},
               {'label': 'FTT-USD', 'value':'FTT-USD'},
               {'label': 'XMR-USD', 'value':'XMR-USD'},
               {'label': 'XLM-USD', 'value':'XLM-USD'},
               {'label': 'CRO-USD', 'value':'CRO-USD'},
               {'label': 'BCH-USD', 'value':'BCH-USD'},
               {'label': 'ALGO-USD', 'value':'ALGO-USD'},
               {'label': 'BTCB-USD', 'value':'BTCB-USD'},
               {'label': 'TONCOIN-USD', 'value':'TONCOIN-USD'},
               {'label': 'LUNA1-USD', 'value':'LUNA1-USD'},
               {'label': 'FLOW-USD', 'value':'FLOW-USD'},
               {'label': 'VET-USD', 'value':'VET-USD'},
               {'label': 'FIL-USD', 'value':'FIL-USD'},
               {'label': 'ICP-USD', 'value':'ICP-USD'},
               {'label': 'APE3-USD', 'value':'APE3-USD'},
               {'label': 'EOS-USD', 'value':'EOS-USD'},
               {'label': 'FRAX-USD', 'value':'FRAX-USD'},
               {'label': 'HBAR-USD', 'value':'HBAR-USD'},
               {'label': 'XTZ-USD', 'value':'XTZ-USD'},
               {'label': 'XCNA-USD', 'value':'XCNA-USD'},
               {'label': 'MANA-USD', 'value':'MANA-USD'},
               {'label': 'SAND-USD', 'value':'SAND-USD'},
               {'label': 'QNT-USD', 'value':'QNT-USD'},
               {'label': 'CHZ-USD', 'value':'CHZ-USD'},
               {'label': 'WBNB-USD', 'value':'WBNB-USD'},
               {'label': 'EGLD-USD', 'value':'EGLD-USD'},
               {'label': 'THETA-USD', 'value':'THETA-USD'},
               {'label': 'AXS-USD', 'value':'AXS-USD'},
               {'label': 'TUSD-USD', 'value':'TUSD-USD'},
               {'label': 'BSV-USD', 'value':'BSV-USD'},
               {'label': 'USDP-USD', 'value':'USDP-USD'},
               {'label': 'OKB-USD', 'value':'OKB-USD'},
               {'label': 'KCS-USD', 'value':'KCS-USD'},
               {'label': 'ZEC-USD', 'value':'ZEC-USD'},
               {'label': 'BTT-USD', 'value':'BTT-USD'},
               {'label': 'XEC-USD', 'value':'XEC-USD'},
               {'label': 'BTT2-USD', 'value':'BTT2-USD'},
               {'label': 'HBTC-USD', 'value':'HBTC-USD'},
               {'label': 'GRT1-USD', 'value':'GRT1-USD'},
               {'label': 'Amazon.com, Inc.', 'value': 'AMZN'},
               {'label': 'Apple Inc.', 'value': 'AAPL'},
               {'label': 'Invitation Homes Inc.', 'value': 'INVH'},
               {'label': 'Tesla, Inc.', 'value': 'TSLA'},
               {'label': 'Advanced Micro Devices, Inc.', 'value': 'AMD'},
               {'label': 'Annaly Capital Management, Inc.', 'value': 'NLY'},
               {'label': 'Intel Corporation', 'value': 'INTC'},
               {'label': 'Rivian Automotive, Inc.', 'value': 'RIVN'},
               {'label': 'AT&T Inc.', 'value': 'T'},
               {'label': 'NIO Inc.', 'value': 'NIO'},
               {'label': 'NVIDIA Corporation', 'value': 'NVDA'},
                {'label': 'NIO Inc.', 'value': 'NIO'},
                {'label': 'Ford Motor Company.', 'value': 'F'},
                {'label': 'Change Healthcare Inc..', 'value': 'NIO'},
                {'label': 'Advanced Micro Devices, Inc..', 'value': 'AMD'},
                {'label': 'Ita?? Unibanco Holding S.A.', 'value': 'SA'},
                {'label': 'Carnival Corporation & plc', 'value': 'CCL'},
                {'label': 'Bank of America Corporation', 'value': 'BAC'},
                {'label': 'American Airlines Group Inc.', 'value': 'AAL'},
                {'label': 'Alphabet Inc.', 'value': 'GOOGL'},
                {'label': 'General Motors Company', 'value': 'GM'},
                {'label': 'Barrick Gold Corporation', 'value': 'GOLD'},
                {'label': 'Alphabet Inc.', 'value': 'GOOG'},

               ]


s = Screener()
data = s.get_screeners('all_cryptocurrencies_us', count=250)
dicts = data['all_cryptocurrencies_us']['quotes']
symbols = [d['symbol'] for d in dicts]

cl = []
for i in symbols:
    cl.append({'label': i, 'value': i})