class Workflow:
    async def __call__(self, problem):
        x = await self.retrieve(problem)
        v = await self.review(x)
        y = await self.generate(v)
        return y
