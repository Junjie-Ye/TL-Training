import json
import os
import re
from copy import deepcopy
from argparse import ArgumentParser


def data_split(file, is_mae=True):
    def sentence_extraction(sentence):
        pat = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
        sentence = pat.sub('', sentence)
        sentence = sentence.lower()

        return sentence

    if is_mae:
        error_calling = ['is not defined', 'positional argument', 'unexpected keyword', '\'related_searches\'', 'ConnectionError', '\"error\":', 'not supported',
                         'Invalid API call', 'Response error', 'Object of type SerpResults is not JSON serializable', '400 Client Error',
                         '\'error\'', 'slice indices must be integers or None or have an __index__ method', 'connection error']
    else:
        error_calling = []

    save_data = []

    with open(file, 'r', encoding='utf8') as f:
        data = json.load(f)
        for sample in data:
            turn = 0
            for index, conv in enumerate(sample["conversations"]):
                if conv["role"] != 'assistant':
                    continue
                turn += 1
                if index != len(sample["conversations"]) - 1:
                    conv_n = sample["conversations"][index+1]
                    if conv_n["role"] != 'function':
                        continue
                    flag = 0
                    for k in error_calling:
                        if sentence_extraction(k) in sentence_extraction(conv_n["content"]):
                            flag = 1
                            break
                    if flag:
                        continue
                save_data.append(
                    {"id": f'Turn {str(turn)}: {sample["id"]}', "conversations": sample["conversations"][:index+1]})

    return save_data


def format_transfor(data, save_file):
    def codellama_process():
        save_data = []

        for sample in data:
            tools = []

            function = json.loads(sample["conversations"][0]["content"])
            for func in function:
                tool = 'Function:\ndef ' + func["function"]["name"] + '('
                parameters = func["function"]["parameters"]
                required = parameters["required"]
                for i, p in enumerate(required):
                    if i != len(required) - 1:
                        tool = tool + p + ': ' + \
                            parameters["properties"][p]["type"] + ', '
                    else:
                        tool = tool + p + ': ' + \
                            parameters["properties"][p]["type"]

                all_param = list(parameters["properties"].keys())
                param = [i for i in all_param if i not in required]
                if len(required) != 0:
                    for p in param:
                        tool = tool + ', ' + p + ': ' + \
                            parameters["properties"][p]["type"] + ' = None'
                else:
                    for i, p in enumerate(param):
                        if i != len(param) - 1:
                            tool = tool + p + ': ' + \
                                parameters["properties"][p]["type"] + \
                                ' = None' + ', '
                        else:
                            tool = tool + p + ': ' + \
                                parameters["properties"][p]["type"] + ' = None'
                tool = tool + '):\n    """\n    ' + \
                    func["function"]["description"] + '\n\n    Args:'
                for p in required:
                    tool = tool + '\n    ' + p + \
                        ' (' + parameters["properties"][p]["type"] + '): ' + \
                        parameters["properties"][p]["description"]
                for p in param:
                    tool = tool + '\n    ' + p + \
                        ' (' + parameters["properties"][p]["type"] + '): ' + \
                        parameters["properties"][p]["description"]

                tool = tool + '\n    """'
                tools.append(tool)

            save_sample = {}
            save_sample["id"] = sample["id"]
            save_sample["conversations"] = [{"role": "system", "content": '\n\n'.join(
                tools)}]
            for conv in sample["conversations"][1:]:
                if conv["role"] in ["user", "function"]:
                    save_sample["conversations"].append(deepcopy(conv))
                else:
                    a, i = conv["content"].split('\n')[0], json.loads(
                        conv["content"].split('\n')[1])
                    a = a + '('
                    param = []
                    for k, v in i.items():
                        param.append(str(k) + '=' +
                                     json.dumps(v, ensure_ascii=False))
                    a = a + ', '.join(param) + ')'
                    save_sample["conversations"].append(
                        {"role": "assistant", "content": a})

            save_data.append(deepcopy(save_sample))

        return save_data

    save_data = codellama_process()
    with open(save_file, 'w', encoding='utf8') as f:
        json.dump(save_data, f, ensure_ascii=False)


def args_parse():
    parser = ArgumentParser()

    parser.add_argument('--file', type=str, default='../../data/raw_data.json')
    parser.add_argument('--save_file', type=str,
                        default='../../data/mae_data.json')
    parser.add_argument('--is_mae', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parse()
    data = data_split(args.file, args.is_mae)
    format_transfor(data, args.save_file)
