def parse_parameters(remainder):
    error = False
    parameters = {}
    if len(remainder) % 2 != 0:
        error = True
    else:
        for i in range(0, len(remainder), 2):
            if not remainder[i].startswith("--"):
                error = True
                break
            else:
                parameters[remainder[i][2:]] = remainder[i + 1]

    if error:
        raise RuntimeError("Invalid parameters specification! Must be of the form: --KEY1 VALUE --KEY2 VALUE2 ...")

    return parameters


def create_override_appendix(keys, parameters):
    override_appendix = ""
    for key in sorted(keys):
        if key in parameters:
            override_appendix += "_" + key + "=" + str(parameters[key])

    return override_appendix
