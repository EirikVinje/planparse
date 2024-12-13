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

    def _generate_prompt(self, context):

        context = copy.deepcopy(context)

        if isinstance(context, dict):
            rendered_context = {
                "text" : context.get("text", ""),
                "label" : context.get("label", ""),
            }
            rendered_template = self.template.render(rendered_context)
        
        elif isinstance(context, str):
            rendered_template = {
                "text" : context
            }
            rendered_template = self.template.render(rendered_template)

        else:
            raise ValueError(f"Invalid context type {type(context)}")
            
        rendered_template = rendered_template.strip()

        return rendered_template
    
    
    def __call__(self, context):
        return self._generate_prompt(context)
        

if __name__ == "__main__":

    template_path = "./prompt_templates/llama/llama_1b_v1.jinja"

    prompter = Prompter(
        template_path = template_path,   
    )
    
    prompter.load()

    # input_text = {"text" : "Dette er et dokument med utnyttingsgrader! æøå", "label" : "BYA"}
    input_text = "Dette er et dokument med utnyttingsgrader! æøå"

    output = prompter(input_text)

    print(output)    
