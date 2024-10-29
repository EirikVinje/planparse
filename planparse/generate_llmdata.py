import os

# import jinja


class PromptFormatter:

    def __init__(
            self,
            templatepath : str = "./prompt_templates/mistral_7b_v1.jinja",
            dir : str = None,
            ):

        """
    
        Makes llm data : prompt instruction + data
        
        :param str templatepath: path to jinja template (llm prompt instruction)
        :param str dir: path to directory with data
        
        """

        pass


    def _load_template(self):
        pass

    
    def _load_data(self):
        pass



    def generate(self):
        pass







if __name__ == "__main__":
    

    formatter = PromptFormatter(dir="./data")

