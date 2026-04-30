# İş Kuralları Motoru (Business Rules Engine):
# Algılanan ham nesneleri (Person, Hardhat vs.) alır,
# İSG kurallarını işleterek mantıksal ihlalleri (NO-Hardhat vb.) hesaplar.

class ViolationEvaluator:
    def evaluate(self, detected_classes: list[str]) -> list[str]:
        violations = []

        is_human_present = any(cls in detected_classes for cls in ["Person", "NO-Hardhat", "NO-Mask", "Face"])

        human_indicators = [
            "Person", "Face", 
            "Hardhat", "Safety Vest", 
            "NO-Hardhat", "NO-Mask", "NO-Safety Vest"
        ]
        is_human_present = any(cls in detected_classes for cls in human_indicators)

        if is_human_present:
            
            # Kural 1: Baret tespit edilmediyse (model uyduruk NO-Hardhat dese de demese de) ihlal.
            if "Hardhat" not in detected_classes:
                violations.append("NO-Hardhat")
                
            # Kural 2: Yelek tespit edilmediyse ihlal.
            if "Safety Vest" not in detected_classes:
                violations.append("NO-Safety Vest")
                
            # Kural 3: Maske tespit edilmediyse ihlal.
            if "Mask" not in detected_classes:
                violations.append("NO-Mask")

            # Kural 4: Gözlük tespit edilmediyse ihlal.
            if "Goggles" not in detected_classes:
                violations.append("NO-Goggles")
                
            # Kural 5: Eldiven tespit edilmediyse ihlal.
            if "Gloves" not in detected_classes:
                violations.append("NO-Gloves")

        return violations