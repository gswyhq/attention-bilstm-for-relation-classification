
import tornado.web
from api.relation_classification_handler import RelClassHandler

class MyFile(tornado.web.StaticFileHandler):

    def set_extra_headers(self, path):
        self.set_header("Cache-control", "no-cache")
        self.set_header("Content-Type", "text/plain; charset=utf-8")  # 若是HTML文件，用浏览器访问时，显示所有的文件内容
        # self.set_header("Content-Type", "text/html; charset=utf-8")  # 若是HTML文件，用浏览器访问时，仅仅显示body部分；

urls = [
        (r'/', RelClassHandler),
        (r'/nlp/classifynet/infer', RelClassHandler),
        (r'/api', RelClassHandler),
        (r"/myfile/(.*)", MyFile, {"path": "./output/"})# 提供静态文件下载； 如浏览器打开‘http://192.168.3.145:8000/myfile/place.pdf’即可访问‘./output/place.pdf’文件
]

# curl http://147.123.19.17:123456/nlp/classifynet/infer -d '{"pid": "adbcd", "question":[["等待期是多久", {}], ["太平洋的面积是多少", {}]]}'
#
# {"data": [[{"score": 0.9857213497161865, "intent": "闲聊"}, {"score": 0.009191920049488544, "intent": "定义"}, {"score": 0.0020747119560837746, "intent": "保障期限"}, {"score": 0.0019243658753111959, "intent": "保全时效"}, {"score": 0.00038407021202147007, "intent": "保全规则_保单还款"}], [{"score": 0.37077027559280396, "intent": "产品卖点"}, {"score": 0.23972906172275543, "intent": "定义"}, {"score": 0.22477547824382782, "intent": "产品价格"}, {"score": 0.03479596599936485, "intent": "闲聊"}, {"score": 0.023998238146305084, "intent": "保全流程"}]], "code": 0}
