import copy 
import json

from jinja2 import Template

class Prompter:
    def __init__(
            self, 
            template_path : str
            ):

        self.template_path = template_path
        
        self.raw_template_text : str = None
        self.template : Template = None
    
    def load(self):

        with open(self.template_path, 'r', encoding='utf-8') as fp:
            self.raw_template_text = fp.read()

        self.template = Template(self.raw_template_text)

    def _generate_prompt(self, context, output):

        context = copy.deepcopy(context)        
        
        rendered_context = {
            "document" : context, 
            "output_text" : output,
            }

        rendered_template = self.template.render(rendered_context)
        
        rendered_template = rendered_template.strip()

        return rendered_template
    
    def __call__(self, context, output):
        return self._generate_prompt(context, output)
        

if __name__ == "__main__":

    template_path = "./prompt_templates/mistral_7b_train_vt.jinja"

    prompter = Prompter(
        template_path = template_path,   
    )
    
    prompter.load()

    input_text = "Dette er et dokument med utnyttingsgrader! æøå"
    output = '{"utnyttingsgrad": ["BYA"]}'

    output = prompter(input_text, output)

    print(output)    
