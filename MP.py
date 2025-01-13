import re

class MP:
    """
    Class MP (Mapping Process) converts genotypes into phenotypes by replacing non-terminal symbols with terminal symbols.
    """

    # Constructor.
    def __init__(self):
        # Production rules.
        self.productions = {
            '<Start>': ['<Expr>'],
            '<Expr>': ['<Expr><Op><Expr>', '<Filter>(<Expr>)', '<Terminal>'],
            '<Filter>': ['<Gau>', '<Arith>', '<Lap>'],
            '<Gau>': ['ft.Gau1', 'ft.Gau2', 'ft.GauDX', 'ft.GauDY'],
            '<Lap>': ['ft.LapG1', 'ft.LapG2', 'ft.Lap'],
            '<Arith>': ['ft.Sqrt', 'ft.Sqr','ft.Log', 'ft.M05', 'ft.Abs', 'ft.Average', 'ft.Median', 'ft.HEq'],
            '<Op>': ['+', '-', '*', '/'],
            '<Terminal>': ['img']
        }

    # Function to expand a non-terminal symbol using the genotype.
    def expand_symbol(self, symbol, genotype, gen_index):
        if symbol in self.productions:
            choices = self.productions[symbol]
            choice_index = genotype[gen_index] % len(choices)
            return choices[int(choice_index)], gen_index + 1
        return symbol, gen_index


    # Main function to generate a string from the start symbol.
    def generate(self, genotype, wr):
        current_string = "<Start>"
        gen_index = 0
        if wr > 1:
            genotype * wr

        # Iterative expansion.
        while True:
            # Search for all non-terminals in the current string.
            non_terminals = re.findall(r'<[^>]+>', current_string)
            if gen_index == (len(genotype) - 1):
                return 'Worst'
            if not non_terminals:
                return current_string

            # Replace the first non-terminal found.
            for non_terminal in non_terminals:
                expansion, gen_index = self.expand_symbol(non_terminal, genotype, gen_index)
                current_string = current_string.replace(non_terminal, expansion, 1)
                break