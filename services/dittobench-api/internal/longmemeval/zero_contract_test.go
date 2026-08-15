package longmemeval

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type zeroEvidenceFixture struct {
	GoEvidenceSHA256 string   `json:"go_evidence_sha256"`
	LatencyMS        uint64   `json:"latency_ms"`
	Evidence         Evidence `json:"evidence"`
}

func buildZeroEvidenceFixture(t *testing.T) []byte {
	t.Helper()
	profile := loadTestProfile(t)
	profile.CasesPerCapability = 2
	selection, err := Select(profile, syntheticCases(profile.CasesPerCapability+5))
	if err != nil {
		t.Fatal(err)
	}
	score, err := Aggregate(selection, balancedOutcomes(selection, 0))
	if err != nil {
		t.Fatal(err)
	}
	providers, err := newRecordingMeter(profile).Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	evidence, err := newAllReceivedFailuresEvidence(
		profile, selection, artifactDigestA, fixtureLatencyMS, score, providers, len(selection.Cases),
	)
	if err != nil {
		t.Fatal(err)
	}
	digest, err := evidence.Digest(profile, selection)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.MarshalIndent(zeroEvidenceFixture{
		GoEvidenceSHA256: digest, LatencyMS: evidence.LatencyMS, Evidence: evidence,
	}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return append(raw, '\n')
}

func TestCommittedOfficialZeroFixtureMatchesGoEvidence(t *testing.T) {
	want := buildZeroEvidenceFixture(t)
	path := filepath.Join("testdata", "go_longmem_official_zero_v2.json")
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v\nwant:\n%s", path, err, want)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s drifted from Go official-zero evidence\nwant:\n%s", path, want)
	}
}

func TestOfficialZeroFixtureRequiresExplicitUniqueEmptyReceiptDigest(t *testing.T) {
	raw := buildZeroEvidenceFixture(t)
	for name, mutate := range map[string]func([]byte) []byte{
		"missing": func(value []byte) []byte {
			return bytes.Replace(
				value,
				[]byte("        \"cost_usd_micros\": 0,\n        \"receipt_set_sha256\": \"\"\n"),
				[]byte("        \"cost_usd_micros\": 0\n"),
				1,
			)
		},
		"duplicate": func(value []byte) []byte {
			return bytes.Replace(
				value,
				[]byte("        \"receipt_set_sha256\": \"\""),
				[]byte("        \"receipt_set_sha256\": \"\",\n        \"receipt_set_sha256\": \"\""),
				1,
			)
		},
	} {
		t.Run(name, func(t *testing.T) {
			changed := mutate(raw)
			if bytes.Equal(changed, raw) {
				t.Fatal("test mutation did not change fixture")
			}
			var decoded zeroEvidenceFixture
			if err := json.Unmarshal(changed, &decoded); err == nil {
				t.Fatal("noncanonical zero receipt digest decoded")
			}
		})
	}
}
