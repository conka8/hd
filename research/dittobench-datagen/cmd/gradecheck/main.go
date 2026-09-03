// gradecheck grades a candidate answer with the OFFICIAL grader.
//
// Our own PASS/FAIL heuristics have now been wrong twice, most damagingly in
// EXP-007 where a probe compared answers against the grader's internal cents
// representation and reported three correct answers as failures. The fix is
// not a better heuristic: it is to stop writing heuristics and call the real
// grader. This binary is evaluation tooling only and is never linked into the
// miner.
//
//	echo '{"cases":[{"case":<MemoryCase>,"response":<RunResponse>}]}' | gradecheck
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/ditto-assistant/dittobench-datagen/grade"
	"github.com/ditto-assistant/dittobench-datagen/protocol"
)

type item struct {
	Case     protocol.MemoryCase  `json:"case"`
	Response protocol.RunResponse `json:"response"`
}

type input struct {
	Cases []item `json:"cases"`
}

func main() {
	var in input
	if err := json.NewDecoder(os.Stdin).Decode(&in); err != nil {
		fmt.Fprintln(os.Stderr, "decode:", err)
		os.Exit(2)
	}
	out := make([]map[string]any, 0, len(in.Cases))
	for _, it := range in.Cases {
		v := grade.Memory(it.Case, it.Response)
		out = append(out, map[string]any{
			"id":       it.Case.ID,
			"type":     it.Case.QuestionType,
			"expected": it.Case.ExpectedAnswer,
			"score":    v.Score,
			"notes":    v.Notes,
		})
	}
	json.NewEncoder(os.Stdout).Encode(out)
}
