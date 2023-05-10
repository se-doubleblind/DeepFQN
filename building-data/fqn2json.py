'''Process outputs from FQNTypeResolution containing node-level FQN rewrites
to construct data instances for DeepFQN.
'''
import json
from json import JSONDecodeError

from pathlib import Path

from tqdm import tqdm

from transformers import RobertaTokenizer


VARIABLE_DECLARATION_NODES = [
    "VARIABLE_DECLARATION_EXPRESSION", "VARIABLE_DECLARATION_FRAGMENT",
    "VARIABLE_DECLARATION_STATEMENT",
]


def validate_node(debug_text, node_text):
    '''org.eclipse.jdt.core.dom.ASTNode.toString() in JDT introduces
    additional newline and space characters. Strip these characters
    to compare whether node extracted from source matches debug text.

    Arguments:
        debug_text (str):
            Debugging text in AST node.
        node_text (str):
            AST Node text extracted from source.

    Returns:
        (bool):
            True, if node text in source matches AST node debug string.
            False, otherwise.
    '''
    debug_text = debug_text.replace('\n', '')
    debug_text = debug_text.replace(' ', '')
    node_text = node_text.replace('\n', '')
    node_text = node_text.replace(' ', '')

    if debug_text == node_text:
        return True
    return False


def process_ast_node(node, method_source, simple_names_mapper):
    '''Process each AST node and depending on TILE algorithm, will insert
    [blank] tokens.

    Arguments:
        node (dict):
            Contains AST node information.
        method_source (str):
            Source code corresponding to entire method.
        simple_names_mapper (dict):
            Maps simple names to corresponding FQNs.

    Returns:
        (tuple)
    '''
    node_type = node['nodeType']
    source = method_source[int(node['start']): int(node['end'])]
    unresolved = node['unresolvedType']
    resolved = node['resolvedType']

    if node_type == "ARRAY_CREATION":
        if resolved is None:
            return None
        if unresolved in simple_names_mapper:
            resolved = simple_names_mapper[unresolved]
        if unresolved == "String":
            full = source.replace("String", "java.lang.String", 1)
            partial = source.replace("String", f"<blank>.String", 1)
            return (
                "java.lang.String", full, partial, "java.lang",
            )
        if unresolved == resolved:
            return None

        full = source.replace(unresolved, resolved, 1)
        partial = source.replace(unresolved, f"<blank>.{unresolved}", 1)
        blank = resolved.replace(f".{unresolved}", "")
        return (
            resolved, full, partial, blank,
        )

    elif node_type == "CAST_EXPRESSION":
        if resolved is None:
            return None
        if unresolved in simple_names_mapper:
            resolved = simple_names_mapper[unresolved]
        if unresolved == resolved:
            return None

        full = source.replace(unresolved, resolved, 1)
        partial = source.replace(unresolved, f"<blank>.{unresolved}", 1)
        blank = resolved.replace(f".{unresolved}", "")
        return (
            resolved, full, partial, blank,
        )

    elif node_type == "CLASS_INSTANCE_CREATION":
        if resolved is None:
            return None
        if unresolved in simple_names_mapper:
            resolved = simple_names_mapper[unresolved]
        if unresolved == "String":
            resolved = "java.lang.String"
        if unresolved == resolved:
            return None

        if unresolved in resolved:
            full = source.replace(unresolved, resolved, 1)
            blank = resolved.replace(f".{unresolved}", "")
        else:
            # For instances, where an error is initialized in a throw statement.
            full = source.replace(unresolved, f"{resolved}.{unresolved}", 1)
            blank = resolved
        partial = source.replace(unresolved, f"<blank>.{unresolved}", 1)
        return (
            resolved, full, partial, blank,
        )

    elif node_type in ["CONSTRUCTOR_INVOCATION", "SUPER_CONSTRUCTOR_INVOCATION"]:
        if resolved is None:
            if unresolved is None:
                return None
        if unresolved == resolved:
            blank = unresolved
        else:
            blank = resolved

        full = node['full']
        partial = node['partial']
        return (
            resolved, full, partial, blank,
        )

    elif node_type in ["FIELD_ACCESS", "SUPER_FIELD_ACCESS"]:
        if resolved is None:
            return None

        if unresolved == resolved:
            return None

        full = node['full']
        partial = node['partial']
        blank = resolved.replace(f"{unresolved}", "")
        return (
            resolved, full, partial, blank,
        )

    elif node_type == "INSTANCEOF_EXPRESSION":
        if resolved is None:
            return None
        if unresolved in simple_names_mapper:
            resolved = simple_names_mapper[unresolved]
        if unresolved == "String":
            resolved = "java.lang.String"
        if unresolved == resolved:
            return None

        full = source.replace(unresolved, resolved, 1)
        partial = source.replace(unresolved, f"<blank>.{unresolved}", 1)
        blank = resolved.replace(f".{unresolved}", "")
        return (
            resolved, full, partial, blank,
        )

    elif node_type in ["METHOD_INVOCATION", "SUPER_METHOD_INVOCATION"]:
        # if node_type == "METHOD_INVOCATION":
        #     if node["validExpression"] == "false":
        #         return None
        if resolved is None:
            if unresolved is None:
                return None
        if unresolved == resolved:
            return None

        partial = "<blank>" + unresolved + node['nodeArguments']
        if unresolved not in resolved:
            full = resolved + unresolved + node['nodeArguments']
            blank = resolved
        else:
            full = resolved + node['nodeArguments']
            blank = resolved.replace(unresolved, '', 1)

        return (
            resolved, full, partial, blank,
        )

    elif node_type == "SINGLE_VARIABLE_DECLARATION":
        if resolved is None:
            return None

        if unresolved == "String":
            full = source.replace("String", "java.lang.String", 1)
            partial = source.replace("String", f"<blank>.String", 1)
            blank = "java.lang"
            return (
                "java.lang.String", full, partial, "java.lang",
            )

        if unresolved == resolved:
            return None

        if len(unresolved.split(' ')) > 1:
            return None

        full = source.replace(unresolved, resolved, 1)
        partial = source.replace(unresolved, f"<blank>.{unresolved}", 1)
        blank = resolved.replace(f".{unresolved}", '', 1)
        return (
            resolved, full, partial, blank,
        )

    elif node_type == "THROW_STATEMENT":
        if 'full' in node:
            full = node['full']
            partial = node['partial']
            blank = node['resolvedType']
            return (
                blank, full, partial, blank,
            )
        return None

    else:
        return None


def process_data_instance(path_to_source, tokenizer):
    '''Process each data instance to construct corresponding data instance
    for DeepFQN model.

    Arguments:
        path_to_source (pathlib.Path):
            Path to source file.
        tokenizer (transformers.RobertaTokenizer):
            Tokenizer.
    '''
    try:
        with open(path_to_source, 'r', encoding='utf-8') as fileobj:
            data = json.load(fileobj)

        method_snippet = data['methodSnippet']
        fqn_method_snippet, hole_method_snippet, hole_method_tokens = [], [], []
        legal_nodes = 0

        legal_nodes, simple_names_mapper = [], {}
        for node in data['nodeInfo']:
            node_start, node_end = int(node['start']), int(node['end'])
            if node_end > len(method_snippet):
                continue

            text =  method_snippet[node_start: node_end]
            try:
                if node['nodeType'] == "SIMPLE_NAME":
                    simple_names_mapper[node['unresolvedType']] = node['resolvedType']
            except KeyError:
                continue
            legal_nodes.append(node)

        if len(legal_nodes) == 0:
            return None

        nodes = sorted(legal_nodes, key=lambda x: (int(x['start']), -int(x['end'])))
        node_tuples = [[nodes[0]]]

        for node in nodes[1:]:
            start, end = int(node['start']), int(node['end'])
            compare_to_node = node_tuples[-1][0]
            ctn_start, ctn_end = int(compare_to_node['start']), int(compare_to_node['end'])
            if start >= ctn_start and end <= ctn_end:
                node_tuples[-1].append(node)
            else:
                node_tuples.append([node])

        assert len(nodes) == sum([len(node_tuple) for node_tuple in node_tuples])

        previous_end_idx, hole_method_length = 0, 0
        blank_idx, resolved_idx = {}, {}

        for i, node_tuple in enumerate(node_tuples):
            preceding_code = method_snippet[previous_end_idx: int(node_tuple[0]['start'])]
            fqn_method_snippet.append(preceding_code)
            hole_method_snippet.append(preceding_code)
            preceding_code_tokens = tokenizer.tokenize(preceding_code)
            hole_method_tokens += preceding_code_tokens
            hole_method_length += len(preceding_code_tokens)

            first_node = {
                'source': method_snippet[int(node_tuple[0]['start']): int(node_tuple[0]['end'])],
                'fqn': method_snippet[int(node_tuple[0]['start']): int(node_tuple[0]['end'])],
                'hole': method_snippet[int(node_tuple[0]['start']): int(node_tuple[0]['end'])],
                'start': int(node_tuple[0]['start']),
                'end': int(node_tuple[0]['end']),
                'blanks': [],
                'resolved': [],
            }
            first_processed = process_ast_node(node_tuple[0], method_snippet, simple_names_mapper)

            if first_processed is not None:
                resolved, fqn_text, hole_text, blank = first_processed
                first_node.update({
                    'fqn': fqn_text,
                    'hole': hole_text,
                    'blanks': [blank],
                    'resolved': [resolved],
                })
                for node in node_tuple[1:]:
                    node_source = method_snippet[int(node['start']): int(node['end'])]
                    processed = process_ast_node(node, method_snippet, simple_names_mapper)
                    if processed is not None:
                        node_resolved, node_fqn_text, node_hole_text, node_blank = processed
                        if (node_source in first_node['fqn']) and (node_source in first_node['hole']):
                            updated_fqn = first_node['fqn'].replace(node_source, node_fqn_text)
                            updated_hole = first_node['hole'].replace(node_source, node_hole_text)
                            first_node.update({
                                'fqn': updated_fqn,
                                'hole': updated_hole,
                                'blanks': first_node['blanks'] + [node_blank],
                                'resolved': first_node['resolved'] + [node_resolved],
                            })

            fqn_method_snippet.append(first_node['fqn'])
            hole_method_snippet.append(first_node['hole'])
            hole_text_tokens = tokenizer.tokenize(first_node['hole'])
            blank_indices = [_id for _id, token in enumerate(hole_text_tokens) \
                             if token == "<blank>"]

            if len(blank_indices) == len(first_node['blanks']):
                for _id, index in enumerate(blank_indices):
                    blank_idx[hole_method_length + index] = first_node['blanks'][_id]
                    resolved_idx[hole_method_length + index] = first_node['resolved'][_id]

            hole_method_tokens += hole_text_tokens
            hole_method_length += len(hole_text_tokens)
            previous_end_idx = first_node['end']

        if previous_end_idx < len(method_snippet):
            fqn_method_snippet.append(method_snippet[previous_end_idx: ])
            hole_method_snippet.append(method_snippet[previous_end_idx: ])
            hole_method_tokens += tokenizer.tokenize(method_snippet[previous_end_idx: ])

        return {
            'path_to_source': data['pathToSource'],
            'code': method_snippet,
            'fqn_code': ''.join(fqn_method_snippet),
            'hole_code': ''.join(hole_method_snippet),
            'hole_tokens': hole_method_tokens,
            'pairs': blank_idx,
            'resolved_pairs': resolved_idx,
        }

    except JSONDecodeError as e:
        print(f"Skipping for {Path(path_to_source).stem}")
        return None


if __name__ == '__main__':
    projects = ['android', 'gwt', 'hibernate-orm', 'joda-time', 'jdk', 'xstream']

    for project in projects:
        path_to_datasets = Path("../../datasets")
#        path_to_project = path_to_datasets / 'raw-fqn' / project
        path_to_project = f"/home/axy190020/research/deepfqn/data/fqndata/typedata/{project}"
        tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')
        tokenizer.add_tokens('<blank>')
        data_instances = []

        id = 0
        for path_to_source in Path(path_to_project).iterdir():
            data_instance = {
                'id': f'{Path(path_to_project).name}-{str(id).zfill(8)}',
                'file': Path(path_to_source).name,
            }
            id += 1
            processed = process_data_instance(Path(path_to_source), tokenizer)
            if processed is not None:
                if processed['pairs']:
                    data_instance.update(processed)
                    data_instances.append(data_instance)

        print(f"Number of data instances: {len(data_instances)}")
        path_to_output = path_to_datasets / 'full-fqn'
        path_to_output.mkdir(exist_ok=True, parents=True)

        with open(f'{path_to_output / project}.json', 'w', encoding='utf-8') as fileobj:
            json.dump(data_instances, fileobj, indent=2)
