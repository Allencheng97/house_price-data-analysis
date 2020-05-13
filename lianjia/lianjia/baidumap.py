import requests
def getlng(address):
    """
    @ address: 名称字符串
    @ 返回值：经度，纬度
    """
    address='山东省威海市'+address
    base_url = "http://api.map.baidu.com/geocoder?address={address}&output=json&key=gmWGXazzhusjfBhAXCWX99hRsuf681GC".format(address=address)
    response = requests.get(base_url)
    answer = response.json()
    latitude = answer['result']['location']['lng']
    longitude = answer['result']['location']['lat']

    return [latitude, longitude]
