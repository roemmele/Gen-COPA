import json


def read_data(params):
    data = []
    with open(params["data_file"]) as f:
        for i, item in enumerate(f):
            item = {'item_id': i, **json.loads(item)}
            item.update({arg: val for arg, val in params.items()
                         if val != None})
            data.append(item)
    return data
