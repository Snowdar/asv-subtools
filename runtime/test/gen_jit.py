import sys
from pathlib import Path
import torch




if __name__ == '__main__':
    f = Path(__file__).resolve()
    proj_path = f.parent.parent.parent.parent
    sub_path = Path.joinpath(proj_path,'subtools/pytorch')
    sys.path.insert(0, str(proj_path))
    sys.path.insert(0, str(sub_path))


    from model import (
        transformer_xvector,
        factored_xvector,
        ecapa_tdnn_xvector,
        resnet_xvector,
        repvgg_xvector,
    )

    target = 5994
    input_dim=80
    model_name = 'init.zip'
    model_path = Path(f.parent) / 'models' / model_name
    quant = False
    m = transformer_xvector.TransformerXvector(input_dim,target,training=False)  # conformer 256D-4H-4Sub
    # m = factored_xvector.Xvector(input_dim,target,training=False)   # ftdnn c1024
    # m = ecapa_tdnn_xvector.ECAPA_TDNN(input_dim,target,training=False) # ecapa c1024
    # m = repvgg_xvector.RepVggXvector(input_dim,target,training=False) # repspk (RepVGG_A1 base32)
    # m = resnet_xvector.ResNetXvector(input_dim,target,training=False) # repnet34  base32

    # Export quantized jit torch script model
    if quant:
        quantized_model = torch.quantization.quantize_dynamic(
            m, {torch.nn.Linear}, dtype=torch.qint8
        )

        script_quant_model = torch.jit.script(quantized_model)
        script_quant_model.save(model_path)
    else:
        m_jit = torch.jit.script(m)
        m_jit.save(model_path)
    m.loss=None
    print(m)
    p = sum(p.numel() for p in m.parameters())
    print("Model params : {}".format(p))
    print("Export model done!")