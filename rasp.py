from dataclasses import dataclass, replace, field
from pprint import pprint
import abc
import typing
import torch


@dataclass(frozen=True)
class SequenceVariableValue:
    type = "sequence"
    value: torch.Tensor


@dataclass(frozen=True)
class MatrixVariableValue:
    type = "matrix"
    value: torch.Tensor


Variable = SequenceVariableValue | MatrixVariableValue


@dataclass(frozen=True)
class ProgramState:
    tokens: torch.Tensor
    sequence_variables: dict[str, SequenceVariableValue] = field(default_factory=dict)
    matrix_variables: dict[str, MatrixVariableValue] = field(default_factory=dict)

    def assign_sequence(
        self, variable_name: str, value: SequenceVariableValue
    ) -> "ProgramState":
        assert (
            variable_name not in self.sequence_variables
            and variable_name not in self.matrix_variables
        )
        return replace(
            self, sequence_variables={**self.sequence_variables, variable_name: value}
        )

    def assign_matrix(
        self, variable_name: str, value: MatrixVariableValue
    ) -> "ProgramState":
        assert (
            variable_name not in self.sequence_variables
            and variable_name not in self.matrix_variables
        )
        return replace(
            self, matrix_variables={**self.matrix_variables, variable_name: value}
        )


T = typing.TypeVar("T", bound=(SequenceVariableValue | MatrixVariableValue))


class Expression(abc.ABC, typing.Generic[T]):
    @abc.abstractmethod
    def evaluate(self, state: ProgramState) -> T:
        pass


SequenceExpression = Expression[SequenceVariableValue]
MatrixExpression = Expression[MatrixVariableValue]

Predicate = typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def eq_predicate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a == b


def lt_predicate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a < b


def gt_predicate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a > b


def leq_predicate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a <= b


def geq_predicate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a >= b


@dataclass(frozen=True)
class SelectExpression(MatrixExpression):
    k: SequenceExpression
    q: SequenceExpression
    p: Predicate

    def evaluate(self, state: ProgramState) -> MatrixVariableValue:
        return MatrixVariableValue(
            value=self.p(
                self.k.evaluate(state).value.unsqueeze(0),
                self.q.evaluate(state).value.unsqueeze(1),
            )
        )


@dataclass(frozen=True)
class SelectorWidthExpression(SequenceExpression):
    m: MatrixExpression

    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue(self.m.evaluate(state).value.sum(dim=1))


@dataclass(frozen=True)
class NotSequenceExpression(SequenceExpression):
    s: SequenceExpression

    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue(~self.s.evaluate(state).value)


@dataclass(frozen=True)
class IndicatorExpression(SequenceExpression):
    s: SequenceExpression

    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue((self.s.evaluate(state).value > 0).float())


@dataclass
class NotMatrixExpression(MatrixExpression):
    m: MatrixExpression

    def evaluate(self, state: ProgramState) -> MatrixVariableValue:
        return MatrixVariableValue(value=~self.m.evaluate(state).value)


@dataclass
class AndMatrixExpression(MatrixExpression):
    a: MatrixExpression
    b: MatrixExpression

    def evaluate(self, state: ProgramState) -> MatrixVariableValue:
        return MatrixVariableValue(
            value=self.a.evaluate(state).value & self.b.evaluate(state).value
        )


@dataclass(frozen=True)
class OrSequenceExpression(SequenceExpression):
    a: SequenceExpression
    b: SequenceExpression

    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue(
            value=self.a.evaluate(state).value | self.b.evaluate(state).value
        )


@dataclass(frozen=True)
class AndSequenceExpression(SequenceExpression):
    a: SequenceExpression
    b: SequenceExpression

    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue(
            value=self.a.evaluate(state).value & self.b.evaluate(state).value
        )


class TrueSequenceExpression(SequenceExpression):
    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue(
            value=torch.ones(state.tokens.shape[0], dtype=torch.bool)
        )


class TrueMatrixExpression(MatrixExpression):
    def evaluate(self, state: ProgramState) -> MatrixVariableValue:
        return MatrixVariableValue(
            value=torch.ones(
                state.tokens.shape[0], state.tokens.shape[0], dtype=torch.bool
            )
        )


class TokensExpression(SequenceExpression):
    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue(value=state.tokens)


class IndicesExpression(SequenceExpression):
    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue(value=torch.arange(state.tokens.shape[0]))


@dataclass
class SequenceReferenceExpression(SequenceExpression):
    variable_name: str

    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return state.sequence_variables[self.variable_name]


@dataclass
class MatrixReferenceExpression(MatrixExpression):
    variable_name: str

    def evaluate(self, state: ProgramState) -> MatrixVariableValue:
        return state.matrix_variables[self.variable_name]


Token = float


@dataclass(frozen=True)
class LiteralSequenceExpression(SequenceExpression):
    literal_value: Token

    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue(
            value=torch.full((state.tokens.shape[0],), self.literal_value)
        )


class LastColumnExpression(SequenceExpression):
    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue(
            value=torch.full((state.tokens.shape[0],), state.tokens.shape[0] - 1)
        )


@dataclass(frozen=True)
class LiteralMatrixExpression(MatrixExpression):
    literal_value: bool

    def evaluate(self, state: ProgramState) -> MatrixVariableValue:
        return MatrixVariableValue(
            value=torch.full(
                (state.tokens.shape[0], state.tokens.shape[0]), self.literal_value
            )
        )


@dataclass(frozen=True)
class AggregateExpression(SequenceExpression):
    m: MatrixExpression
    s: SequenceExpression

    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        s_value = self.s.evaluate(state).value
        m_value = self.m.evaluate(state).value
        mult = s_value * m_value
        agg = mult.sum(dim=1)
        avg = agg / m_value.sum(dim=1)
        return SequenceVariableValue(value=avg)


@dataclass(frozen=True)
class CompareSequencesExpression(SequenceExpression):
    a: SequenceExpression
    b: SequenceExpression
    p: Predicate

    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue(
            value=self.p(
                self.a.evaluate(state).value,
                self.b.evaluate(state).value,
            )
        )


@dataclass(frozen=True)
class CompareMatricesExpression(MatrixExpression):
    a: MatrixExpression
    b: MatrixExpression
    p: Predicate

    def evaluate(self, state: ProgramState) -> MatrixVariableValue:
        return MatrixVariableValue(
            value=self.p(
                self.a.evaluate(state).value,
                self.b.evaluate(state).value,
            )
        )


@dataclass(frozen=True)
class SubtractSequencesExpression(SequenceExpression):
    a: SequenceExpression
    b: SequenceExpression

    def evaluate(self, state: ProgramState) -> SequenceVariableValue:
        return SequenceVariableValue(
            value=self.a.evaluate(state).value - self.b.evaluate(state).value
        )


@dataclass(frozen=True)
class Statement:
    assignment_name: str
    expression: SequenceExpression | MatrixExpression


@dataclass(frozen=True)
class Program:
    statements: tuple[Statement, ...]

    def invoke(self, tokens: torch.Tensor):
        state = ProgramState(tokens=tokens)
        for statement in self.statements:
            result = statement.expression.evaluate(state)
            if isinstance(result, SequenceVariableValue):
                state = state.assign_sequence(statement.assignment_name, result)
            elif isinstance(result, MatrixVariableValue):
                state = state.assign_matrix(statement.assignment_name, result)
            else:
                raise NotImplementedError()
        print(result)


double_histogram_program = Program(
    statements=(
        Statement(
            "same_tok",
            SelectExpression(TokensExpression(), TokensExpression(), eq_predicate),
        ),
        Statement(
            "hist", SelectorWidthExpression(MatrixReferenceExpression("same_tok"))
        ),
        Statement(
            "is_prev",
            SelectExpression(IndicesExpression(), IndicesExpression(), lt_predicate),
        ),
        Statement(
            "has_prev",
            CompareSequencesExpression(
                SelectorWidthExpression(
                    AndMatrixExpression(
                        MatrixReferenceExpression("is_prev"),
                        MatrixReferenceExpression("same_tok"),
                    ),
                ),
                LiteralSequenceExpression(0),
                gt_predicate,
            ),
        ),
        Statement(
            "first", NotSequenceExpression(SequenceReferenceExpression("has_prev"))
        ),
        Statement(
            "same_count",
            SelectExpression(
                SequenceReferenceExpression("hist"),
                SequenceReferenceExpression("hist"),
                eq_predicate,
            ),
        ),
        Statement(
            "first_mat",
            SelectExpression(
                SequenceReferenceExpression("first"),
                TrueSequenceExpression(),
                eq_predicate,
            ),
        ),
        Statement(
            "same_count_reprs",
            AndMatrixExpression(
                MatrixReferenceExpression("same_count"),
                MatrixReferenceExpression("first_mat"),
            ),
        ),
        Statement(
            "hist2",
            SelectorWidthExpression(MatrixReferenceExpression("same_count_reprs")),
        ),
    )
)

double_histogram_program.invoke(torch.tensor([1, 1, 1, 2, 2, 3, 3, 4, 5, 6]))


def frac_prevs(sop: SequenceExpression, val: Token) -> SequenceExpression:
    prevs = SelectExpression(IndicesExpression(), IndicesExpression(), leq_predicate)
    return AggregateExpression(
        prevs,
        IndicatorExpression(
            CompareSequencesExpression(
                sop, LiteralSequenceExpression(val), eq_predicate
            )
        ),
    )


def pair_balance(open_token: Token, close_token: Token) -> SequenceExpression:
    opens = frac_prevs(TokensExpression(), open_token)
    closes = frac_prevs(TokensExpression(), close_token)
    return SubtractSequencesExpression(opens, closes)


OPEN_A_TOKEN: Token = 1.0
CLOSE_A_TOKEN: Token = 2.0
OPEN_B_TOKEN: Token = 3.0
CLOSE_B_TOKEN: Token = 4.0

shuffle_dyck_2_program = Program(
    statements=(
        Statement("bal1", pair_balance(OPEN_A_TOKEN, CLOSE_A_TOKEN)),
        Statement("bal2", pair_balance(OPEN_B_TOKEN, CLOSE_B_TOKEN)),
        Statement(
            "negative",
            OrSequenceExpression(
                CompareSequencesExpression(
                    SequenceReferenceExpression("bal1"),
                    LiteralSequenceExpression(0),
                    lt_predicate,
                ),
                CompareSequencesExpression(
                    SequenceReferenceExpression("bal2"),
                    LiteralSequenceExpression(0),
                    lt_predicate,
                ),
            ),
        ),
        Statement(
            "had_neg",
            CompareSequencesExpression(
                AggregateExpression(
                    TrueMatrixExpression(),
                    IndicatorExpression(SequenceReferenceExpression("negative")),
                ),
                LiteralSequenceExpression(0),
                gt_predicate,
            ),
        ),
        Statement(
            "select_last",
            SelectExpression(IndicesExpression(), LastColumnExpression(), eq_predicate),
        ),
        Statement(
            "end_0",
            AggregateExpression(
                MatrixReferenceExpression("select_last"),
                AndSequenceExpression(
                    CompareSequencesExpression(
                        SequenceReferenceExpression("bal1"),
                        LiteralSequenceExpression(0),
                        eq_predicate,
                    ),
                    CompareSequencesExpression(
                        SequenceReferenceExpression("bal2"),
                        LiteralSequenceExpression(0),
                        eq_predicate,
                    ),
                ),
            ),
        ),
        Statement(
            "shuffle_dyck_2",
            AndSequenceExpression(
                CompareSequencesExpression(
                    SequenceReferenceExpression("end_0"),
                    LiteralSequenceExpression(1),
                    eq_predicate,
                ),
                NotSequenceExpression(SequenceReferenceExpression("had_neg")),
            ),
        ),
    )
)


shuffle_dyck_2_program.invoke(
    torch.tensor([OPEN_A_TOKEN, OPEN_B_TOKEN, CLOSE_A_TOKEN, CLOSE_B_TOKEN])
)


shuffle_dyck_2_program.invoke(
    torch.tensor(
        [OPEN_A_TOKEN, OPEN_B_TOKEN, CLOSE_A_TOKEN, CLOSE_B_TOKEN, CLOSE_B_TOKEN]
    )
)
