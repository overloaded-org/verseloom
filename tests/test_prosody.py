import unittest
from wordconv_lab.prosody import all_templates, audit_tones, generate_template, variants


class ProsodyGrammarTests(unittest.TestCase):
    def test_four_five_character_reference_tables(self):
        expected = {
            ("P",False): ("PPPZZ","ZZZPP","ZZPPZ","PPZZP","PPPZZ","ZZZPP","ZZPPZ","PPZZP"),
            ("P",True): ("PPZZP","ZZZPP","ZZPPZ","PPZZP","PPPZZ","ZZZPP","ZZPPZ","PPZZP"),
            ("Z",False): ("ZZPPZ","PPZZP","PPPZZ","ZZZPP","ZZPPZ","PPZZP","PPPZZ","ZZZPP"),
            ("Z",True): ("ZZZPP","PPZZP","PPPZZ","ZZZPP","ZZPPZ","PPZZP","PPPZZ","ZZZPP"),
        }
        for (start,rhymed), lines in expected.items():
            self.assertEqual(generate_template(5,start,rhymed).core,lines)

    def test_all_eight_templates_have_correct_relations(self):
        for n in (5,7):
            for template in all_templates(n):
                self.assertEqual(len(template.core),8)
                for row,line in enumerate(template.core):
                    self.assertEqual(len(line),n)
                    self.assertEqual(line[-1],"P" if row in template.rhyme_rows else "Z")
                for a,b in ((0,1),(2,3),(4,5),(6,7)):
                    self.assertNotEqual(template.core[a][1],template.core[b][1])
                for a,b in ((1,2),(3,4),(5,6)):
                    self.assertEqual(template.core[a][1],template.core[b][1])

    def test_first_rhyme_does_not_leak_into_fifth_line(self):
        for n in (5,7):
            for start in ("P","Z"):
                t=generate_template(n,start,True)
                self.assertEqual(t.core[0][-1],"P")
                self.assertEqual(t.core[4][-1],"Z")

    def test_seven_character_known_start(self):
        self.assertEqual(generate_template(7,"P",False).core[:4],
                         ("PPZZPPZ","ZZPPZZP","ZZPPPZZ","PPZZZPP"))

    def test_safe_prefix_does_not_allow_unrescued_isolated_ping(self):
        self.assertNotIn("ZPZZP",variants("PPZZP"))
        self.assertIn("PZPPZ",variants("ZZPPZ"))
        self.assertIn("PZZPPZZ",variants("ZZPPPZZ"))

    def test_unknown_reading_not_inferred_from_meter(self):
        lines=list(generate_template(5,"P",False).core)
        lines[0]="?"+lines[0][1:]
        self.assertEqual(audit_tones(lines)["status"],"needs_reading_review")

    def test_real_annotated_variant_outside_profile_is_not_called_invalid(self):
        # Zeng Yongyi's annotated Du Shenyan example has a PPP->PZZZ tail variant.
        tones=("ZZZPP","PPZZP","PPZZZ","PZZPP","ZZPPZ","PPZZP","ZPPZZ","PZZPP")
        result=audit_tones(tones)
        self.assertEqual(result["status"],"outside_supported_profile")
        self.assertIn("NOT mean",result["warning"])

    def test_dufu_annotated_example_matches_safe_profile(self):
        tones=("ZZPPZZP","PPZZZPP","ZPZZPPZ","ZZPPZZP","PZZPPZZ","ZPPZZPP","ZPZZPPZ","PZPPZZP")
        self.assertEqual(audit_tones(tones)["matches"],["L7-Z-R1"])
