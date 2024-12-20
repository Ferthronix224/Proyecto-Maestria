import re

class MP:
    """
    Class MP (Mapping Process) converts genotypes into phenotypes by replacing non-terminal symbols with terminal symbols.
    """
    def __init__(self):
        # Definición de las reglas de producción
        self.productions = {
            '<Start>': ['<Expr>'],
            '<Expr>': ['<Expr><Op><Expr>', '<Filter>(<Expr>)', '<Terminal>'],
            '<Filter>': ['<Gau>', '<Arith>', '<Lap>'],
            '<Gau>': ['ft.Gau1', 'ft.Gau2'],
            '<Lap>': ['ft.LapG1', 'ft.LapG2', 'ft.Lap'],
            '<Arith>': ['ft.Sqrt', 'ft.Log'],
            '<Op>': ['+', '-', '*', '/'],
            '<Terminal>': ['img']
        }

    # Función para expandir un símbolo no terminal usando el genotipo
    def expand_symbol(self, symbol, genotype, gen_index):
        if symbol in self.productions:
            choices = self.productions[symbol]
            choice_index = genotype[gen_index] % len(choices)
            return choices[int(choice_index)], gen_index + 1
        return symbol, gen_index


    # Función principal para generar una cadena desde el símbolo de inicio
    def generate(self, genotype, wr):
        current_string = "<Start>"
        gen_index = 0
        if wr > 1:
            genotype * wr

        # Expansión iterativa
        while True:
            # Buscar todos los no terminales en la cadena actual
            non_terminals = re.findall(r'<[^>]+>', current_string)
            if gen_index == (len(genotype) - 1):
                return 'Worst'
            if not non_terminals:
                return current_string

            # Reemplazar el primer no terminal encontrado
            for non_terminal in non_terminals:
                expansion, gen_index = self.expand_symbol(non_terminal, genotype, gen_index)
                current_string = current_string.replace(non_terminal, expansion, 1)
                break  # Salir del bucle para reiniciar la búsqueda de no terminales