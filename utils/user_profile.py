# utils/user_profile.py

class UserProfile:
    def __init__(self):
        self.age = None
        self.income = None
        self.location = None
        self.category = None
        self.occupation = None
        self.education = None
        self.is_complete = False

    def set_profile(self, age, income, location, category, occupation="", education=""):
        self.age = age
        self.income = income
        self.location = location
        self.category = category
        self.occupation = occupation
        self.education = education
        self.is_complete = True

    def get_profile_string(self):
        if not self.is_complete:
            return ""

        profile = f"Age: {self.age}, Income: {self.income}, Location: {self.location}"
        if self.category:
            profile += f", Category: {self.category}"
        if self.occupation:
            profile += f", Occupation: {self.occupation}"
        if self.education:
            profile += f", Education: {self.education}"
        return profile

    def get_search_context(self):
        """Get formatted context for better search results"""
        if not self.is_complete:
            return ""

        context_parts = []
        if self.category and self.category.lower() != 'general':
            context_parts.append(self.category)
        if self.location:
            context_parts.append(self.location)
        if self.age:
            try:
                age_num = int(self.age)
                if age_num < 18:
                    context_parts.append("minor student")
                elif age_num < 25:
                    context_parts.append("young adult student")
                elif age_num > 60:
                    context_parts.append("senior citizen elderly")
            except:
                pass

        return " ".join(context_parts)