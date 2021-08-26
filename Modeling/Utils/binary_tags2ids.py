def binary_tags2ids(_input, output, id2tag_dict, istrain=False):
    if torch.cuda.is_available():
        _input = _input.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
    else:
        _input = _input.detach().numpy()
        output = output.detach().numpy()

    to_dict_id = lambda x: [id2tag_dict[_x] for _x in x]

    if not istrain:
        output -= _input

    tags_idxes = output.argsort(axis=1)[:, ::-1][:, :10]

    return list(map(to_dict_id, tags_idxes))