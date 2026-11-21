"""Разрезает склеенные тензоры Qwen 3.5 в GGUF, чтобы файл загружался в Ollama.

Новые версии llama.cpp (в том числе та, что ставит Unsloth в Colab) пишут для qwen35
склеенные по выходным строкам тензоры:
    blk.N.attn_qkv_gate.weight  = [attn_qkv | attn_gate]
    blk.N.ssm_beta_alpha.weight = [ssm_beta | ssm_alpha]
llama.cpp внутри Ollama 0.40 ждёт их раздельно и падает с
"check_tensor_dims: tensor 'blk.0.ssm_beta.weight' not found".

Квантованные блоки лежат построчно, поэтому строки режутся побайтно:
без переквантования и без потери качества.

Запуск (gguf-py из клона llama.cpp):
    uv run --no-project --with numpy --with <llama.cpp>/gguf-py python -I \
        backend/train/gguf_unfuse_qwen35.py <вход.gguf> <выход.gguf>
"""

import sys

import gguf
import numpy as np

# Склеенный тензор -> (первая часть, вторая часть, размер первой части по выходу).
# Размер первой части берётся из соседних полей модели, см. split_sizes().
FUSED = {"attn_qkv_gate": ("attn_qkv", "attn_gate"), "ssm_beta_alpha": ("ssm_beta", "ssm_alpha")}


def field_int(reader, key):
    f = reader.fields[key]
    return int(f.parts[f.data[0]][0])


def split_sizes(reader):
    arch = "qwen35"
    v_heads = field_int(reader, f"{arch}.ssm.time_step_rank")  # Число V-голов линейного внимания.
    inner = field_int(reader, f"{arch}.ssm.inner_size")  # Размер выхода (gate) = v_heads * head_v_dim.
    return {"attn_qkv_gate": None, "ssm_beta_alpha": v_heads, "_gate": inner}


def main(src, dst):
    reader = gguf.GGUFReader(src)
    arch_field = reader.fields[gguf.Keys.General.ARCHITECTURE]
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode()
    if arch != "qwen35":
        sys.exit(f"ожидалась архитектура qwen35, а в файле {arch}")
    sizes = split_sizes(reader)

    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    plan = []  # (имя, данные, тип) в порядке записи.
    for t in reader.tensors:
        stem = t.name.rsplit(".", 1)[0]  # blk.N.attn_qkv_gate
        kind = stem.split(".", 2)[-1]
        if kind in FUSED:
            first, second = FUSED[kind]
            rows = t.data.shape[0]
            n_first = rows - sizes["_gate"] if kind == "attn_qkv_gate" else sizes[kind]
            if not 0 < n_first < rows:
                sys.exit(f"{t.name}: не удалось определить разрез ({rows} строк, первая часть {n_first})")
            prefix = stem.rsplit(".", 1)[0]
            plan.append((f"{prefix}.{first}.weight", t.data[:n_first], t.tensor_type))
            plan.append((f"{prefix}.{second}.weight", t.data[n_first:], t.tensor_type))
        else:
            plan.append((t.name, t.data, t.tensor_type))

    for name, data, ttype in plan:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, ttype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, data, _ in plan:
        writer.write_tensor_data(np.ascontiguousarray(data), tensor_endianess=reader.endianess)
    writer.close()
    split = sum(1 for n, _, _ in plan if n.endswith((".attn_gate.weight", ".ssm_alpha.weight")))
    print(f"готово: {len(reader.tensors)} тензоров -> {len(plan)}, разрезано {split}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
