class Workflow:
    async def __call__(self, problem):
        x = await self.retrieve(problem)
        y = await self.generate(x)
        return y
