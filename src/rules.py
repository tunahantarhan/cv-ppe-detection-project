# İş Kuralları Motoru (Business Rules Engine):
# Algılanan ham nesneleri (Person, Hardhat vs.) alır,
# İSG kurallarını işleterek mantıksal ihlalleri (NO-Hardhat vb.) hesaplar.

class ViolationEvaluator:
    def evaluate(self, detected_classes: list[str]) -> list[str]:
        violations = []

        if "NO-Hardhat" in detected_classes:
            violations.append("Baret_Yok")
            
        if "NO-Safety Vest" in detected_classes:
            violations.append("Is_Yelegi_Yok")

        return violations