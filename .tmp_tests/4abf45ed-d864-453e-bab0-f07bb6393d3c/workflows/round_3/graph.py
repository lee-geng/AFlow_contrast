class Workflow:
    async def __call__(self, problem):
        y = await self.generate(problem)
        return y
