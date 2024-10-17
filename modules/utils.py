from jinja2 import Environment, meta


def get_template_args(template):
    args = meta.find_undeclared_variables(Environment().parse(template))
    return args


def instantiate_template(item, template):
    text = Environment().from_string(template).render(**dict(item.items()), zip=zip)
    return text
