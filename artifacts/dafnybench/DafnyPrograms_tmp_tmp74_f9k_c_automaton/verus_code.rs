use vstd::prelude::*;

verus! {
    /**
    Consider cellular automata: a row of cells is repeatedly updated according to a rule. In this exercise I dabbled with,
    each cell has the value either false or true. Each cell's next state depends only on the immediate neighbours, in the 
    case where the cell is at the edges of the row, the inexistent neighbours are replaced by "false". The automaton table 
    will contain the initial row, plus a row for each number of steps.
     */

    /**
    This method computes the automaton.
    Provide the initial row: init, the rule and the desired number of steps
     */
    fn execute_automaton(init: Vec<bool>, rule: fn(bool, bool, bool) -> bool, steps: usize) -> (table: Vec<Vec<bool>>)
        requires 
            // we need the initial row to have the length bigger or equal to two
            init.len() >= 2,
        ensures 
            // after computation the automaton is made of the initial row plus a row for each of the steps
            table.len() == 1 + steps,
            // the automaton must have the initial row at the top
            table[0] == init,
            // all rows in the automaton must be the same length
            forall|i: int| 0 <= i < table.len() ==> table[i].len() == init.len(),
            // all the middle row elements (with existing neighbours) after a step, will be equal to the rule applied on the element in the previous state
            // and its neigbours
            forall|i: int| 0 <= i < table.len() - 1 ==>
              forall|j: int| 1 <= j <= table[i].len() - 2 ==> 
                table[i + 1][j] == rule(table[i][j - 1], table[i][j], table[i][j + 1]),
            // the corner row elements (with non-existing neighbours) after a step, will be equal to the rule applied on the element in the previous state,
            // its neighbour and false
            forall|i: int| 0 <= i < table.len() - 1 ==>
              table[i + 1][0] == rule(false, table[i][0], table[i][1]) && 
              table[i + 1][table[i].len() - 1] == rule(table[i][table[i].len() - 2], table[i][table[i].len() - 1], false),
    {
        let mut result: Vec<Vec<bool>> = vec![init.clone()];
        let mut current_seq = init;
        let mut index = 0;

        while index < steps
            invariant 
                0 <= index <= steps,
                result.len() == index + 1,
                result[0] == init,
                current_seq.len() == init.len(),
                forall|i: int| 0 <= i < result.len() ==> result[i].len() == init.len(),
                current_seq == result[index as int],
                forall|i: int| 0 <= i < result.len() - 1 ==>
                  forall|j: int| 1 <= j <= result[i].len() - 2 ==> 
                    result[i + 1][j] == rule(result[i][j - 1], result[i][j], result[i][j + 1]),
                forall|i: int| 0 <= i < result.len() - 1 ==>
                  result[i + 1][0] == rule(false, result[i][0], result[i][1]) && 
                  result[i + 1][result[i].len() - 1] == rule(result[i][result[i].len() - 2], result[i][result[i].len() - 1], false),
            decreases steps - index,
        {
            let mut next_seq: Vec<bool> = Vec::new();
            
            // First element (left edge)
            next_seq.push(rule(false, current_seq[0], current_seq[1]));
            
            // Middle elements  
            let mut j = 1;
            while j < current_seq.len() - 1
                invariant
                    current_seq.len() == init.len(),
                    1 <= j <= current_seq.len() - 1,
                    next_seq.len() == j,
                    next_seq[0] == rule(false, current_seq[0], current_seq[1]),
                    forall|k: int| 1 <= k < j ==> next_seq[k] == rule(current_seq[k - 1], current_seq[k], current_seq[k + 1]),
                decreases current_seq.len() - j,
            {
                next_seq.push(rule(current_seq[j - 1], current_seq[j], current_seq[j + 1]));
                j = j + 1;
            }
            
            // Last element (right edge)
            next_seq.push(rule(current_seq[current_seq.len() - 2], current_seq[current_seq.len() - 1], false));
            
            current_seq = next_seq.clone();
            result.push(next_seq);
            index = index + 1;
        }

        result
    }

    // example rule
    fn the_rule(a: bool, b: bool, c: bool) -> bool {
        a || b || c
    }

    // example rule 2
    fn the_rule2(a: bool, b: bool, c: bool) -> bool {
        a && b && c
    }

    fn test_method() {
        // the initial row
        let init = vec![false, false, true, false, false];

        // calculate automaton steps with 
        let result = execute_automaton(init.clone(), the_rule, 3);
        // the intial row plus the three steps of the automaton are showed bellow
        assert(result[0] == vec![false, false, true, false, false]); // the initial state of the automaton
        assert(result[1] == vec![false, true, true, true, false]); // after the first step
        assert(result[2] == vec![true, true, true, true, true]); // after the second step
        assert(result[3] == vec![true, true, true, true, true]); // after the third step, remains the same from now on

        let result2 = execute_automaton(init, the_rule2, 2);
        // the intial row plus the two steps of the automaton are showed bellow
        assert(result2[0] == vec![false, false, true, false, false]); // the initial state of the automaton
        assert(result2[1] == vec![false, false, false, false, false]); // after the first step
        assert(result2[2] == vec![false, false, false, false, false]); // after the second step, remains the same from now on
    }
}

fn main() {}