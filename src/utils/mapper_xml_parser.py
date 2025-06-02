import os
from typing import List, Dict, Any
from lxml import etree

def parse_mapper_xml(xml_path: str) -> List[Dict[str, Any]]:
    """
    MyBatis 등 Java XML 매퍼 파일에서 쿼리/매핑 정보를 추출합니다.
    반환: [ { 'id': str, 'type': select|insert|update|delete, 'sql': str, ... } ]
    """
    if not os.path.exists(xml_path):
        return []
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    try:
        root = etree.fromstring(xml_content.encode('utf-8'))
    except Exception:
        return []
    result = []
    for tag in ['select', 'insert', 'update', 'delete']:
        for elem in root.findall(f'.//{tag}'):
            entry = {
                'id': elem.get('id'),
                'type': tag,
                'parameterType': elem.get('parameterType'),
                'resultType': elem.get('resultType'),
                'sql': ''.join(elem.itertext()).strip(),
            }
            result.append(entry)
    return result 